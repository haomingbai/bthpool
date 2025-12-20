/**
 * @file bthpool.hpp
 * @brief Header for a lightweight, scalable thread pool.
 *
 * Provides `bthpool::detail::BThreadPool`, a configurable thread pool with
 * fast/slow task queues, adaptive worker creation up to a maximum cap, and
 * graceful shutdown semantics. Designed for high-throughput, latency-sensitive
 * workloads and general-purpose asynchronous task execution.
 *
 * Key Features:
 * - Dual-queue scheduling: fast queue (bounded) + slow queue (fallback).
 * - Adaptive worker management: grows up to `max_thread_num` as needed.
 * - Cooperative shutdown: `join()` for graceful, `shutdown()` for immediate.
 * - Portable CPU detection; Linux and Windows support.
 *
 * Thread-Safety:
 * - Public submission APIs are thread-safe.
 * - Lifecycle methods (`join`, `shutdown`, `restart`) synchronize internally.
 *
 * Usage Sketch:
 * @code
 *   BThreadPool pool;              // or BThreadPool(BThreadPoolParam{})
 *   pool.post([]{  work ; });  // fire-and-forget void task
 *   pool.post([]{ return 42; });  // non-void result is discarded
 *   pool.join();                   // wait for completion (graceful)
 * @endcode
 *
 * @author  Haoming Bai <haomingbai@hotmail.com>
 * @date    2025-12-07
 * @version 0.1.0
 * @copyright Copyright © 2025 Haoming Bai
 * @license  MIT
 * @see      include/bthpool/internal/safe_queue.hpp
 */

#include <algorithm>
#include <atomic>
#ifdef USE_BOOST_ASIO_EXECUTOR
#include <boost/asio/execution_context.hpp>
#endif
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "internal/safe_queue.hpp"

#ifndef __linux__
#include "windows.h"
#else
#include <unistd.h>
#endif

namespace bthpool::detail {

inline size_t get_nproc() noexcept {
#ifndef __linux__
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  return static_cast<size_t>(sysInfo.dwNumberOfProcessors);
#else
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? static_cast<size_t>(n) : 1;
#endif
}

struct BThreadPoolParam {
  size_t core_thread_num{get_nproc()};
  size_t max_thread_num{std::numeric_limits<int>::max()};
  size_t fast_queue_capacity{0};
  size_t thread_clean_interval{60000};
  size_t task_scan_interval{100};
  std::size_t suspend_time{1};
};

class BThreadPool
#ifdef USE_BOOST_ASIO_EXECUTOR
    : public boost::asio::execution_context
#endif
{
 public:
  // Rule-of-five: define destructor; forbid copy/move
  BThreadPool(const BThreadPool&) = delete;
  BThreadPool(BThreadPool&&) noexcept = delete;
  BThreadPool& operator=(const BThreadPool&) = delete;
  BThreadPool& operator=(BThreadPool&&) noexcept = delete;
  ~BThreadPool() {
    shutdown();
  }

  /**
   * @brief Constructs a BThreadPool with default parameters and initializes
   * internal queues.
   *
   * @details
   * - Initializes the configuration parameters (`param_`) to their defaults.
   * - Creates a slow work queue with default capacity.
   * - Creates a fast work queue sized according to
   * `param_.fast_queue_capacity`.
   * - Sets the pool state to RUNNING and resets the living thread count to
   * zero.
   *
   * The constructor does not start any worker threads by itself. Depending on
   * the library design, threads may be started lazily on first task submission
   * or via an explicit start method (e.g., `start()`), if available.
   *
   * @note
   * - If `param_.fast_queue_capacity` is small, high-frequency tasks may be
   * throttled.
   * - Consider tuning `param_` before submitting tasks if your workload is
   * skewed toward fast or slow tasks.
   * - Ensure proper synchronization when interacting with the pool from
   * multiple threads.
   *
   * @usage
   * Example usage outline:
   * 1. Create the pool: `BThreadPool pool;`
   * 2. Optionally configure parameters before starting (e.g., queue capacities,
   * thread counts).
   * 3. Start the pool if required by the API.
   * 4. Submit tasks to fast or slow queues as appropriate for their
   * latency/priority needs.
   * 5. Gracefully shut down the pool (e.g., `stop()` or `join()`), ensuring all
   * tasks complete.
   */
  BThreadPool()
      : param_(),
        slow_queue_(),
        fast_queue_(param_.fast_queue_capacity),
        stat_(RUNNING),
        living_thread_num_(0) {}

  /**
   * @brief Constructs a BThreadPool with the specified parameters.
   *
   * Initializes internal fast and slow work queues, sets the initial pool state
   * to RUNNING, and prepares the pool for task submission. The fast queue is
   * configured with the capacity provided in the parameter, while the slow
   * queue starts unbounded or with its default capacity. No worker threads are
   * started until the pool is fully initialized; thread creation typically
   * occurs when the pool is started or tasks are enqueued depending on
   * implementation details.
   *
   * @param param Configuration for the thread pool, including queue capacities,
   *              concurrency limits, and scheduling behavior. The fast queue
   *              capacity is derived from this parameter.
   *
   * @note After construction, the pool is in a RUNNING state, but the number of
   *       living threads is initially zero. Ensure you start or submit tasks to
   *       spawn workers according to your usage pattern.
   * @note Thread safety: Constructing the pool is not thread-safe and should be
   *       done by a single thread. Subsequent operations on the pool depend on
   *       the class's concurrency guarantees.
   * @throws No exceptions are thrown during construction under normal
   * conditions, but dependent types may throw if invalid parameters are
   * provided.
   *
   * @usage
   *   BThreadPoolParam param;
   *   param.fast_queue_capacity = 1024;  // Set queue capacity and other fields
   *   // ... configure additional parameters ...
   *
   *   // Create the pool
   *   BThreadPool pool(param);
   *
   *   // Submit tasks / start the pool according to the API
   *   // pool.submit([] {  `work`  });
   *   // pool.start(); // if explicit start is required
   *
   * @see BThreadPoolParam for available configuration fields.
   */
  BThreadPool(BThreadPoolParam param)
      : param_(param),
        slow_queue_(),
        fast_queue_(param_.fast_queue_capacity),
        stat_(RUNNING),
        living_thread_num_(0) {}

  /**
   * @brief Schedule a task that returns void for execution by the thread pool.
   *
   * This overload accepts any callable and its arguments that form a
   * void-returning invocable. The callable and arguments are perfectly
   * forwarded, internally bound into a single function object, and then posted
   * to the thread pool.
   *
   * @tparam F Callable type. Must be invocable with Args... and return void.
   * @tparam Args Argument types to pass to the callable.
   * @param f The callable to execute. Can be a function, lambda, functor, or
   * member function bound with placeholders.
   * @param args Arguments to forward to the callable.
   *
   * @note No future or result is provided because the task returns void.
   * @note The callable will be executed asynchronously by the thread pool.
   * @warning Exceptions thrown by the callable are not captured via a future in
   * this overload. Ensure your callable handles exceptions appropriately or the
   * thread pool provides an exception propagation mechanism.
   * @pre `std::invoke(std::decay_t<F>, std::decay_t<Args>...)` is valid and
   * returns void.
   *
   * Usage:
   *   // Post a simple void task
   *   pool.post([] { do_work(); });
   *
   *   // Post a task with arguments
   *   pool.post([](int x, std::string s) { process(x, s); }, 42, "data");
   *
   *   // Post a member function by binding the instance
   *   MyWorker worker;
   *   pool.post([&worker]{ worker.run(); });
   */
  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<std::is_void_v<Ret>, void> post(F&& f, Args&&... args) {
    auto func_ptr = new ThreadFunc(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    post(func_ptr);
  }

  /**
   * @brief Enqueue a callable with arguments for asynchronous execution,
   * discarding its return value.
   *
   * This overload of `post` accepts a callable `F` and arguments `Args...`
   * where the callable produces a non-void result type `Ret`. The result is
   * intentionally ignored, making it suitable for fire-and-forget tasks whose
   * side effects are the only concern.
   *
   * Internally, the callable and its arguments are captured, invoked via
   * `std::apply`, and the return value is discarded. The task is wrapped and
   * forwarded to the thread pool's scheduling mechanism.
   *
   * @tparam F    Callable type to be executed.
   * @tparam Args Argument types to be forwarded to the callable.
   * @tparam Ret  Deduced non-void return type of the callable invocation.
   *
   * @param f     Callable to be executed asynchronously.
   * @param args  Arguments to be forwarded to the callable.
   *
   * @note This function only accepts callables with non-void return types; use
   * the corresponding `post` overload for callables that return `void`.
   * @note Exceptions thrown by the callable will propagate according to the
   * thread pool's handling in `post(ThreadFunc*)`.
   *
   * @par Usage
   * - Use this function for tasks where the result is not needed:
   *   - Logging, metrics updates, notifications, or cache warming.
   * - If you need the result of the computation, prefer a submission method
   * that returns a future or provides a callback to capture the value.
   *
   * @warning The callable and its arguments are captured by value/move; ensure
   * any referenced resources remain valid or are properly owned at the time of
   *          task execution.
   */
  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<!std::is_void_v<Ret>, void> post(F&& f, Args&&... args) {
    // Capture callable and arguments, execute, and discard the return value
    auto func_ptr = new ThreadFunc([fn = std::forward<F>(f),
                                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      (void)std::apply(
          [&](auto&&... xs) -> Ret { return std::invoke(fn, std::forward<decltype(xs)>(xs)...); },
          std::move(tup));
    });
    post(func_ptr);
  }

  /**
   * @brief Enqueue a task to the slow queue for deferred execution.
   *
   * This overload mirrors the `post` API but routes tasks directly to the slow
   * queue, useful for lower-priority or backlog-friendly work. The callable is
   * bound with its arguments and the return value (if any) is discarded.
   *
   * @tparam F Callable type. Must be invocable with Args... and return void.
   * @tparam Args Argument types forwarded to the callable.
   * @param f Callable to execute asynchronously on the slow queue.
   * @param args Arguments to forward to the callable.
   */
  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<std::is_void_v<Ret>, void> defer(F&& f, Args&&... args) {
    auto func_ptr = new ThreadFunc(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    defer(func_ptr);
  }

  /**
   * @brief Enqueue a non-void task to the slow queue, discarding its result.
   *
   * Like the void overload, this submits work to the slow queue for deferred
   * processing. The callable's return value is intentionally ignored.
   *
   * @tparam F Callable type.
   * @tparam Args Argument types forwarded to the callable.
   * @tparam Ret Deduced non-void return type of the callable.
   * @param f Callable to execute asynchronously on the slow queue.
   * @param args Arguments to forward to the callable.
   */
  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<!std::is_void_v<Ret>, void> defer(F&& f, Args&&... args) {
    auto func_ptr = new ThreadFunc([fn = std::forward<F>(f),
                                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      (void)std::apply(
          [&](auto&&... xs) -> Ret { return std::invoke(fn, std::forward<decltype(xs)>(xs)...); },
          std::move(tup));
    });
    defer(func_ptr);
  }

  /**
   * @brief Dispatch a callable for execution, either immediately or via the pool queue.
   *
   * If the caller thread is one of the pool's worker threads, the callable is invoked
   * immediately (inline) on the calling thread. Otherwise, the callable is enqueued
   * for execution by the pool (equivalent to calling @c post()).
   *
   * @tparam F    Callable type.
   * @tparam Args Argument types forwarded to the callable.
   * @param f     The callable to execute.
   * @param args  Arguments to pass to the callable.
   *
   * @note Because execution may occur immediately, any side effects happen before
   *       @c dispatch() returns when called from a worker thread.
   */
  template <typename F, typename... Args>
  void dispatch(F&& f, Args&&... args) {
    // Try to execute the task directly.
    auto curr_tid = std::this_thread::get_id();
    bool is_in_pool = false;
    {
      std::lock_guard lock(map_mtx_);
      is_in_pool = thread_map_.contains(curr_tid);
    }
    if (is_in_pool) {
      std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    } else {
      post(std::forward<F>(f), std::forward<Args>(args)...);
    }
  }

  /**
   * @brief Submit a task and get a `std::future` for its result.
   *
   * Usage:
   *  - `auto fut = pool.futured_post([]{ return 42; });`
   *  - `auto fut = pool.futured_post([](int x){ return x+1; }, 1);`
   *  - `auto fut = pool.futured_post([]{});` // future<void>
   *
   * Returns a future corresponding to the callable's return type.
   * If the callable returns `void`, the type is `std::future<void>`.
   *
   * Notes:
   *  - Exceptions thrown inside the task are captured and set on the future.
   *  - Callable and args are captured by move; ensure lifetimes are
   * appropriate.
   */
  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<std::is_void_v<Ret>, std::future<void>> futured_post(F&& f, Args&&... args) {
    auto promise = std::make_shared<std::promise<void>>();
    auto fut = promise->get_future();
    // Wrap task to fulfill promise regardless of success/failure.
    auto func_ptr = new ThreadFunc([promise, fn = std::forward<F>(f),
                                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      try {
        std::apply([&](auto&&... xs) { std::invoke(fn, std::forward<decltype(xs)>(xs)...); },
                   std::move(tup));
        promise->set_value();
      } catch (...) {
        // Propagate exception to the future.
        promise->set_exception(std::current_exception());
      }
    });
    post(func_ptr);
    return fut;
  }

  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<!std::is_void_v<Ret>, std::future<Ret>> futured_post(F&& f, Args&&... args) {
    auto promise = std::make_shared<std::promise<Ret>>();
    auto fut = promise->get_future();
    auto func_ptr = new ThreadFunc([promise, fn = std::forward<F>(f),
                                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
      try {
        Ret result = std::apply(
            [&](auto&&... xs) -> Ret { return std::invoke(fn, std::forward<decltype(xs)>(xs)...); },
            std::move(tup));
        promise->set_value(std::move(result));
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
    post(func_ptr);
    return fut;
  }

  /**
   * @brief Gracefully shuts down the thread pool by stopping worker threads and
   * waiting for all tasks to complete.
   *
   * This function transitions the pool state to stopping, notifies all worker
   * threads, and joins each thread. It blocks until all queued and in-progress
   * tasks have finished executing and all threads have terminated, then marks
   * the pool as stopped.
   *
   * Thread Safety:
   * - This method is thread-safe; it uses internal synchronization to manage
   * state and thread shutdown.
   * - Should typically be called once, e.g., from the owner thread during
   * destruction or shutdown.
   *
   * Side Effects:
   * - Signals all worker threads to stop after completing their current tasks.
   * - Blocks the calling thread until all workers have exited.
   *
   * Preconditions:
   * - The thread pool must have been started successfully.
   *
   * Postconditions:
   * - No worker threads remain active.
   * - No tasks are left pending or running.
   * - Pool state is set to STOPPED.
   *
   * Usage:
   * - Call join() during application shutdown to ensure all submitted tasks
   * finish:
   *   // Ensure graceful shutdown and completion of all tasks.
   *   // pool.join();
   *
   * Performance Notes:
   * - Blocking duration depends on the number and duration of outstanding
   * tasks.
   */
  void join() {
    stat_.store(STOPPING);
    {
      std::lock_guard<std::mutex> lock(map_mtx_);
      cv_.notify_all();
      std::for_each(thread_map_.begin(), thread_map_.end(), [](auto& p) mutable {
        p.second->set_stop();
        p.second->join();
      });
      thread_map_.clear();
    }
    stat_.store(STOPPED);
  }

  /**
   * @brief Shuts down the thread pool immediately.
   *
   * Sets the internal status to STOPPED, notifies all worker threads, and joins
   * them. This operation does NOT wait for queued or currently running tasks to
   * complete.
   *
   * Usage:
   *  - Call when the pool should stop accepting and executing further work.
   *  - After shutdown, the thread pool instance should not be reused for
   * submitting tasks.
   *
   * Notes:
   *  - Pending tasks may be discarded or left incomplete.
   *  - Running tasks may be interrupted depending on worker stop semantics.
   *  - Ensure any external synchronization or resource cleanup is done prior to
   * invoking this.
   */
  void shutdown() {
    stat_.store(STOPPED);
    // Clean the queues and release resources.
    ThreadFuncPtr func_ptr;
    while (fast_queue_.pop(func_ptr)) {
      delete func_ptr;
    }
    while (slow_queue_.pop(func_ptr)) {
      delete func_ptr;
    }
    {
      std::lock_guard<std::mutex> lock(map_mtx_);
      cv_.notify_all();
      std::for_each(thread_map_.begin(), thread_map_.end(), [](auto& p) mutable {
        p.second->set_stop();
        p.second->join();
      });
      thread_map_.clear();
    }
  }

  /**
   * Restart the thread pool if it is not currently running.
   *
   * This method transitions the internal status from STOPPED to RUNNING using
   * an atomic compare-and-exchange loop with acquire-release semantics to
   * ensure proper synchronization across threads. If the pool is already
   * RUNNING, the call is a no-op.
   *
   * Thread-safety: Safe to call concurrently; only one caller will perform the
   * transition, others will observe RUNNING and return.
   */
  void restart() {
    if (stat_.load() == RUNNING) {
      return;
    }
    Status stat = STOPPED;
    while (!stat_.compare_exchange_weak(stat, RUNNING, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      if (stat == RUNNING) {
        break;
      } else {
        stat = STOPPED;
      }
    }
  }

#ifdef USE_BOOST_ASIO_EXECUTOR
  class BThreadPoolExecutor {
   public:
    BThreadPoolExecutor(const BThreadPoolExecutor&) noexcept = default;
    BThreadPoolExecutor(BThreadPoolExecutor&&) noexcept = default;
    BThreadPoolExecutor(BThreadPool* pool) : pool_(pool) {}
    BThreadPoolExecutor& operator=(const BThreadPoolExecutor&) noexcept = default;
    BThreadPoolExecutor& operator=(BThreadPoolExecutor&&) noexcept = default;

    template <typename... Args>
    void post(Args&&... args) {
      pool_->post(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void defer(Args&&... args) {
      pool_->defer(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void dispatch(Args&&... args) {
      pool_->dispatch(std::forward<Args>(args)...);
    }

    BThreadPool& context() noexcept {
      return *pool_;
    }

   private:
    BThreadPool* pool_;
  };

  BThreadPoolExecutor get_executor() noexcept {
    return BThreadPoolExecutor(this);
  }
#endif

 private:
  using ThreadFunc = std::move_only_function<void()>;
  using ThreadFuncPtr = ThreadFunc*;

  void post(ThreadFuncPtr func_ptr) {
    auto curr_num = living_thread_num_.load(std::memory_order_acquire);
    while (curr_num < param_.core_thread_num) {
      if (living_thread_num_.compare_exchange_weak(
              curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Get the lock.
        // Create a thread to execute the task immediately.
        auto worker_ptr = std::make_unique<ThreadWorker>(this);
        ThreadWorker::run(std::move(worker_ptr));
        break;
      }
    }
    // Push the task to the queue.
    if (fast_queue_.push(func_ptr)) {
      cv_.notify_one();
      return;
    } else {
      curr_num = living_thread_num_.load(std::memory_order_acquire);
      // Create a new thread when all threads are occupied and new threads
      // are available.
      while (curr_num < param_.max_thread_num) {
        if (living_thread_num_.compare_exchange_weak(
                curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          // Get the lock.
          // Create a thread to execute the task immediately.
          auto worker_ptr = std::make_unique<ThreadWorker>(this);
          ThreadWorker::run(std::move(worker_ptr));
          break;
        }
      }
      // Push the task into the slow queue to wait.
      slow_queue_.push(func_ptr);
      cv_.notify_one();
      return;
    }
  }

  void defer(ThreadFuncPtr func_ptr) {
    // If the thread is less than expected, then create a new one.
    auto curr_num = living_thread_num_.load(std::memory_order_acquire);
    while (curr_num < param_.core_thread_num) {
      if (living_thread_num_.compare_exchange_weak(
              curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Get the lock.
        // Create a thread to execute the task immediately.
        auto worker_ptr = std::make_unique<ThreadWorker>(this);
        ThreadWorker::run(std::move(worker_ptr));
        break;
      }
    }
    // Push the task into the slow queue to wait directly.
    slow_queue_.push(func_ptr);
    cv_.notify_one();
  }

  class ThreadWorker {
   public:
    static void run(std::unique_ptr<ThreadWorker> self) noexcept;

    explicit ThreadWorker(BThreadPool* pool) : pool_(pool), should_stop_(false) {}

    ThreadWorker(const ThreadWorker&) = delete;
    ThreadWorker(ThreadWorker&&) noexcept = delete;
    ThreadWorker& operator=(const ThreadWorker&) = delete;
    ThreadWorker& operator=(ThreadWorker&&) noexcept = delete;

    void join() noexcept {
      std::lock_guard<std::mutex> lock(mtx_);
      if (thread_.joinable()) {
        thread_.join();
      }
    }

    ~ThreadWorker() {
      set_stop();
      join();
    }

    bool should_stop() const noexcept {
      return should_stop_.load();
    }

    void set_stop() noexcept {
      should_stop_.store(true);
    }

   private:
    std::mutex mtx_;
    ThreadFunc func_;
    std::thread thread_;
    BThreadPool* const pool_;
    std::atomic<bool> should_stop_;
  };

  class ThreadCleaner {
   public:
    void operator()() {
      worker_->set_stop();
      worker_->join();
      worker_.reset();
    }

    ThreadCleaner(std::unique_ptr<ThreadWorker> worker) : worker_(std::move(worker)) {}

    ThreadCleaner(ThreadCleaner&&) noexcept = default;
    ThreadCleaner& operator=(ThreadCleaner&&) noexcept = default;
    ThreadCleaner(const ThreadCleaner&) noexcept = delete;
    ThreadCleaner& operator=(const ThreadCleaner&) noexcept = delete;

   private:
    std::unique_ptr<ThreadWorker> worker_;
  };

  class ThreadWorkerFunctor {
   public:
    explicit ThreadWorkerFunctor(BThreadPool* pool, ThreadWorker* worker)
        : pool_(pool), worker_(worker), curr_unscanned_time_(0) {}

    void operator()() noexcept {
      for (;;) {
        if (pool_->stat_.load(std::memory_order_acquire) == STOPPED && worker_->should_stop()) {
          // Normal exit, no nead to clean because in the join function,
          // all threads are auto joinned.
          break;
        }
        // Determine whether the pool should scan from the slow queue.
        if (curr_unscanned_time_ >= pool_->param_.thread_clean_interval) {
          // Pop a task from the slow queue.
          ThreadFuncPtr func;
          if (pool_->slow_queue_.pop(func)) {
            curr_unscanned_time_ = 0;
            execute_and_delete_function(func);
          }
        }
        // Get the task from fast queue.
        ThreadFuncPtr func = try_get_task();
        if (func) {
          execute_and_delete_function(func);
        } else {
          // If the thread pool needs joinning,
          // then exit.
          if (pool_->stat_.load(std::memory_order_acquire) == STOPPING) {
            // Normal exit, no nead to clean because in the join function,
            // all threads are auto joinned.
            break;
          }
          if (try_cleanup()) {
            // If cleaned by a cleaner, than the thread should be joinned.
            break;
          }
          // Sleep for a while to wait for a new task.
          std::unique_lock<std::mutex> lock(pool_->mtx_);
          pool_->cv_.wait(lock);
        }
      }
    }

   private:
    void execute_and_delete_function(ThreadFuncPtr func) const noexcept {
      // Check whether the function is a null ptr.
      if (func) {
        try {
          (*func)();
        } catch (...) {
          // Ignore exceptions.
        }
        delete func;
      }
    }

    ThreadFuncPtr try_get_task() {
      ThreadFuncPtr func = nullptr;
      // Try to first
      auto succ = pool_->fast_queue_.pop(func);
      if (succ) {
        return func;
      } else if ((succ = pool_->slow_queue_.pop(func))) {
        // Reset the scanner task of the slow queue.
        curr_unscanned_time_ = 0;
        return func;
      }
      return nullptr;
    }

    bool try_cleanup() {
      if (pool_->stat_.load(std::memory_order_acquire) != RUNNING) {
        return true;
      }
      std::ptrdiff_t curr_num = pool_->living_thread_num_.load(std::memory_order_acquire);
      while (curr_num > pool_->param_.core_thread_num) {
        if (pool_->living_thread_num_.compare_exchange_weak(
                curr_num, curr_num - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          // Successfully get the lock.
          std::lock_guard<std::mutex> lock(pool_->map_mtx_);
          auto tid = std::this_thread::get_id();
          auto it = pool_->thread_map_.find(tid);
          if (it != pool_->thread_map_.end()) {
            // convert unique_ptr to shared_ptr for copyable lambda
            auto worker_ptr(std::move(it->second));
            pool_->thread_map_.erase(it);
            // Cleanup in the next stage with copyable shared_ptr
            ThreadFuncPtr cleaner_ptr = new ThreadFunc([worker = std::move(worker_ptr)]() mutable {
              if (worker) {
                worker->set_stop();
                worker->join();
                worker.reset();
              }
            });
            pool_->post(cleaner_ptr);
          } else {
            return true;
          }
        }
      }
      // The number of thread is same as or lower than the core thread number.
      // No need to clean.
      return false;
    }
    // The context of the thread function.
    BThreadPool* const pool_;
    ThreadWorker* const worker_;
    // Determine the behavior of the thread pool.
    std::size_t curr_unscanned_time_;
  };

  // Parameter of the thread pool.
  const BThreadPoolParam param_;

  // Task queues, including a fast queue and a slow queue.
  SafeQueue<ThreadFuncPtr> slow_queue_;
  LockfreeFixedQueue<ThreadFuncPtr> fast_queue_;

  // Thread map, which can find the thread worker and clean.
  std::mutex map_mtx_;
  std::unordered_map<std::thread::id, std::unique_ptr<ThreadWorker>> thread_map_;

  // Lock and conditional variable
  std::condition_variable cv_;
  std::mutex mtx_;

  // Determine whether the pool should stop.
  enum Status : unsigned char { RUNNING, STOPPING, STOPPED };
  std::atomic<Status> stat_;

  // Indicate the number of working thread.
  std::atomic<std::ptrdiff_t> living_thread_num_;

  // (moved param_ above queues for correct initialization ordering)
};

inline void BThreadPool::ThreadWorker::run(std::unique_ptr<ThreadWorker> self) noexcept {
  self->func_ = ThreadWorkerFunctor{self->pool_, self.get()};
  self->thread_ = std::thread(std::move(self->func_));
  auto tid = self->thread_.get_id();
  {
    std::lock_guard<std::mutex> lock(self->pool_->map_mtx_);
    self->pool_->thread_map_.emplace(tid, std::move(self));
  }
}

}  // namespace bthpool::detail

namespace bthpool {
using BThreadPool = detail::BThreadPool;
}
