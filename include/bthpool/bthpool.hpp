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
 * @version 0.4.1
 * @copyright Copyright © 2025 Haoming Bai
 * @license  MIT
 * @see      include/bthpool/internal/safe_queue.hpp
 */

#pragma once
#ifndef BTHPOOL_BTHPOOL_HPP_
#define BTHPOOL_BTHPOOL_HPP_

#ifdef USE_BOOST_ASIO_EXECUTOR
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/execution.hpp>
#include <boost/asio/execution_context.hpp>
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "internal/safe_queue.hpp"

namespace bthpool::detail {
struct BThreadPoolParam {
  size_t core_thread_num{std::thread::hardware_concurrency()};
  size_t max_thread_num{std::numeric_limits<int>::max()};
  size_t fast_queue_capacity{0};
  size_t thread_clean_interval{60000};
  size_t task_scan_interval{100};
  std::size_t suspend_time{1};
};

template <typename Allocator, typename = void>
struct is_thread_pool_allocator : std::false_type {};

template <typename Allocator>
struct is_thread_pool_allocator<
    Allocator,
    std::void_t<typename std::allocator_traits<Allocator>::value_type,
                typename std::allocator_traits<Allocator>::template rebind_alloc<std::byte>>>
    : std::bool_constant<std::is_copy_constructible_v<Allocator> &&
                         std::is_default_constructible_v<Allocator>> {};

template <typename Allocator = std::allocator<std::byte>>
class BThreadPool
#ifdef USE_BOOST_ASIO_EXECUTOR
    : public boost::asio::execution_context
#endif
{
 private:
  static_assert(is_thread_pool_allocator<Allocator>::value,
                "Allocator must satisfy allocator_traits and be copy/default constructible.");

  class ThreadWorker;
  using allocator_type = Allocator;

  template <typename T>
  using RebindAllocator = typename std::allocator_traits<allocator_type>::template rebind_alloc<T>;
  template <typename T>
  using RebindAllocatorTraits = std::allocator_traits<RebindAllocator<T>>;

  struct ThreadWorkerDeleter {
    allocator_type allocator{};
    void operator()(ThreadWorker* ptr) const noexcept;
  };
  using ThreadWorkerPtr = std::unique_ptr<ThreadWorker, ThreadWorkerDeleter>;

  // C++20 does not provide std::move_only_function. packaged_task keeps
  // move-only callable support while preserving the queue's ownership model.
  using ThreadFunc = std::packaged_task<void()>;
  using ThreadFuncPtr = ThreadFunc*;
  using ThreadFuncAllocator = RebindAllocator<ThreadFunc>;
  using SlowQueueContainer = std::deque<ThreadFuncPtr, RebindAllocator<ThreadFuncPtr>>;
  using SlowQueueType = SafeQueue<ThreadFuncPtr, SlowQueueContainer>;
  using FastQueueType = LockfreeFixedQueue<ThreadFuncPtr, RebindAllocator<ThreadFuncPtr>>;
  using WorkerContainer = std::vector<ThreadWorkerPtr, RebindAllocator<ThreadWorkerPtr>>;

 public:
  using pool_allocator_type = allocator_type;

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
  BThreadPool() : BThreadPool(BThreadPoolParam{}, allocator_type{}) {}

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
   *   // pool.post(task);
   *   // pool.start(); // if explicit start is required
   *
   * @see BThreadPoolParam for available configuration fields.
   */
  BThreadPool(BThreadPoolParam param, allocator_type alloc = allocator_type{})
      : param_(std::move(param)),
        allocator_(std::move(alloc)),
        slow_queue_(SlowQueueContainer{RebindAllocator<ThreadFuncPtr>(allocator_)}),
        fast_queue_(param_.fast_queue_capacity, RebindAllocator<ThreadFuncPtr>(allocator_)),
        stat_(RUNNING),
        workers_(RebindAllocator<ThreadWorkerPtr>(allocator_)),
        live_worker_num_(0),
        idle_worker_num_(0),
        pending_task_num_(0),
        fast_queue_size_(0),
        slow_queue_size_(0) {}

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
    auto func_ptr = make_thread_func(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    if (!post(func_ptr)) {
      destroy_thread_func(func_ptr);
    }
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
    auto func_ptr = make_thread_func(
        [fn = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          (void)std::apply(
              [&](auto&&... xs) -> Ret {
                return std::invoke(fn, std::forward<decltype(xs)>(xs)...);
              },
              std::move(tup));
        });
    if (!post(func_ptr)) {
      destroy_thread_func(func_ptr);
    }
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
    auto func_ptr = make_thread_func(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    if (!defer(func_ptr)) {
      destroy_thread_func(func_ptr);
    }
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
    auto func_ptr = make_thread_func(
        [fn = std::forward<F>(f), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
          (void)std::apply(
              [&](auto&&... xs) -> Ret {
                return std::invoke(fn, std::forward<decltype(xs)>(xs)...);
              },
              std::move(tup));
        });
    if (!defer(func_ptr)) {
      destroy_thread_func(func_ptr);
    }
  }

  /**
   * @brief Dispatch a callable for execution, either immediately or via the
   * pool queue.
   *
   * If the caller thread is one of the pool's worker threads, the callable is
   * invoked immediately (inline) on the calling thread. Otherwise, the callable
   * is enqueued for execution by the pool (equivalent to calling @c post()).
   *
   * @tparam F    Callable type.
   * @tparam Args Argument types forwarded to the callable.
   * @param f     The callable to execute.
   * @param args  Arguments to pass to the callable.
   *
   * @note Because execution may occur immediately, any side effects happen
   * before
   *       @c dispatch() returns when called from a worker thread.
   */
  template <typename F, typename... Args>
  void dispatch(F&& f, Args&&... args) {
    // Try to execute the task directly.
    auto curr_tid = std::this_thread::get_id();

    // Judge the current thread is in pool.
    bool is_in_pool = dispatch_depth_ != 0 && current_pool_ == this;

    // Limit the max recursion depth to avoid stack overflow by counting the
    // thread
    if (is_in_pool && dispatch_depth_ < kMaxDispatchDepth) {
      struct DispatchDepthGuard {
        std::size_t& depth;
        explicit DispatchDepthGuard(std::size_t& d) : depth(d) {
          ++depth;
        }
        ~DispatchDepthGuard() {
          --depth;
        }
      } guard(dispatch_depth_);
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
    auto promise =
        std::allocate_shared<std::promise<void>>(RebindAllocator<std::promise<void>>(allocator_));
    auto fut = promise->get_future();
    // Wrap task to fulfill promise regardless of success/failure.
    auto func_ptr =
        make_thread_func([promise, fn = std::forward<F>(f),
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
    if (!post(func_ptr)) {
      try {
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("thread pool is stopping; task dropped")));
      } catch (...) {
      }
      destroy_thread_func(func_ptr);
    }
    return fut;
  }

  template <typename F, typename... Args,
            typename Ret = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
  std::enable_if_t<!std::is_void_v<Ret>, std::future<Ret>> futured_post(F&& f, Args&&... args) {
    auto promise =
        std::allocate_shared<std::promise<Ret>>(RebindAllocator<std::promise<Ret>>(allocator_));
    auto fut = promise->get_future();
    auto func_ptr = make_thread_func([promise, fn = std::forward<F>(f),
                                      tup =
                                          std::make_tuple(std::forward<Args>(args)...)]() mutable {
      try {
        Ret result = std::apply(
            [&](auto&&... xs) -> Ret { return std::invoke(fn, std::forward<decltype(xs)>(xs)...); },
            std::move(tup));
        promise->set_value(std::move(result));
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
    if (!post(func_ptr)) {
      try {
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("thread pool is stopping; task dropped")));
      } catch (...) {
      }
      destroy_thread_func(func_ptr);
    }
    return fut;
  }

  /**
   * @brief Gracefully shuts down the thread pool by stopping worker threads and
   * waiting for all tasks to complete.
   *
   * This function transitions the pool into a draining state, waits for all
   * accepted work to leave the system, then wakes and joins the workers before
   * marking the pool as stopped.
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
    std::vector<ThreadWorkerPtr> workers;

    reap_exited_workers();

    {
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
      std::unique_lock<std::mutex> lock(state_mtx_);
      if (stat_ == STOPPED) {
        return;
      }

      // Reason: `join()` now flips the pool into a single draining state under
      // one mutex, so new submissions cannot race with shutdown bookkeeping.
      // Effect: accepted tasks finish exactly once and workers observe the same
      // stop condition instead of juggling multiple counters/semaphores.
      stat_ = JOINING;
      state_cv_.notify_all();

      // Ordering matters here: we wait for the accepted-task count to reach
      // zero before moving worker ownership out of `workers_`. Doing it in the
      // opposite order would let a worker touch a destroyed/moved record while
      // it is still finishing its last task.
      state_cv_.wait(lock, [this] { return pending_task_num_ == 0; });

      stat_ = STOPPING;
      state_cv_.notify_all();
      workers.reserve(workers_.size());
      for (auto& worker : workers_) {
        if (worker) {
          workers.emplace_back(std::move(worker));
        }
      }
      workers_.clear();
      fast_queue_size_ = 0;
      slow_queue_size_ = 0;
    }

    join_workers(workers);

    {
      std::lock_guard<std::mutex> lock(state_mtx_);
      stat_ = STOPPED;
      live_worker_num_ = 0;
      idle_worker_num_ = 0;
      state_cv_.notify_all();
    }
  }

  /**
   * @brief Shuts down the thread pool immediately.
   *
   * Stops accepting new tasks, discards queued-but-not-running tasks, wakes all
   * workers, and joins them. Running tasks are allowed to finish their current
   * callable, but no further queued work is drained.
   *
   * Usage:
   *  - Call when the pool should stop accepting and executing further work.
   *  - After shutdown, the thread pool instance should not be reused for
   * submitting tasks.
   *
   * Notes:
   *  - Queued tasks are discarded.
   *  - Running tasks are not interrupted mid-call; they finish and then exit.
   *  - Ensure any external synchronization or resource cleanup is done prior to
   * invoking this.
   */
  void shutdown() {
    std::vector<ThreadWorkerPtr> workers;

    reap_exited_workers();

    {
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
      std::lock_guard<std::mutex> lock(state_mtx_);
      if (stat_ == STOPPED) {
        return;
      }

      stat_ = STOPPING;

      // Reason: queued work must be destroyed while holding the same mutex that
      // guards queue sizes and lifecycle state.
      // Effect: `shutdown()` no longer races with workers draining tasks, so it
      // cannot underflow the pending count or leave futures waiting forever.
      //
      // The sequence is intentional:
      // 1. stop accepting new work
      // 2. drop queued-but-not-running work
      // 3. wake workers so in-flight tasks can finish and exit
      //
      // Reordering these steps reintroduces the old deadlock windows where one
      // side thinks work still exists and the other side already consumed the
      // corresponding wake-up.
      clear_queued_tasks_locked();
      state_cv_.notify_all();

      workers.reserve(workers_.size());
      for (auto& worker : workers_) {
        if (worker) {
          workers.emplace_back(std::move(worker));
        }
      }
      workers_.clear();
    }

    join_workers(workers);

    {
      std::lock_guard<std::mutex> lock(state_mtx_);
      stat_ = STOPPED;
      live_worker_num_ = 0;
      idle_worker_num_ = 0;
      fast_queue_size_ = 0;
      slow_queue_size_ = 0;
      state_cv_.notify_all();
    }
  }

  /**
   * Restart the thread pool if it is not currently running.
   *
   * This method only transitions the pool from STOPPED back to RUNNING. It
   * does not recreate workers eagerly; workers are started lazily by the next
   * accepted submission, which keeps restart cheap and keeps lifecycle logic in
   * one place.
   *
   * Thread-safety: lifecycle methods are serialized by `lifecycle_mtx_`, so a
   * restart cannot interleave with join/shutdown halfway through their stop
   * sequence.
   */
  void restart() {
    reap_exited_workers();

    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
    std::lock_guard<std::mutex> lock(state_mtx_);
    if (stat_ == STOPPED) {
      stat_ = RUNNING;
    }
  }

  class BThreadPoolExecutor {
   public:
#ifdef USE_BOOST_ASIO_EXECUTOR
    using context_type = boost::asio::execution_context;
#else
    using context_type = BThreadPool;
#endif
    using allocator_type = typename BThreadPool::pool_allocator_type;

    BThreadPoolExecutor() noexcept : pool_(nullptr), allocator_() {}
    explicit BThreadPoolExecutor(BThreadPool* pool) noexcept
        : pool_(pool), allocator_(pool ? pool->get_allocator() : allocator_type{}) {}
    BThreadPoolExecutor(const BThreadPoolExecutor&) noexcept = default;
    BThreadPoolExecutor(BThreadPoolExecutor&&) noexcept = default;
    BThreadPoolExecutor& operator=(const BThreadPoolExecutor&) noexcept = default;
    BThreadPoolExecutor& operator=(BThreadPoolExecutor&&) noexcept = default;
    ~BThreadPoolExecutor() = default;

    allocator_type get_allocator() const noexcept {
      return allocator_;
    }

    void on_work_started() const noexcept {}

    void on_work_finished() const noexcept {}

    template <typename F, typename ExecutorAllocator>
    void dispatch(F&& f, const ExecutorAllocator&) const {
      assert(pool_);
      pool_->dispatch(std::forward<F>(f));
    }

    template <typename F, typename ExecutorAllocator>
    void post(F&& f, const ExecutorAllocator&) const {
      assert(pool_);
      pool_->post(std::forward<F>(f));
    }

    template <typename F, typename ExecutorAllocator>
    void defer(F&& f, const ExecutorAllocator&) const {
      assert(pool_);
      pool_->defer(std::forward<F>(f));
    }

    template <typename F>
    void execute(F&& f) const {
      assert(pool_);
      pool_->post(std::forward<F>(f));
    }

#ifdef USE_BOOST_ASIO_EXECUTOR
    boost::asio::any_io_executor to_any_io_executor() const {
      assert(pool_);
      return boost::asio::any_io_executor(*this);
    }

    friend context_type& query(const BThreadPoolExecutor& ex,
                               boost::asio::execution::context_t) noexcept {
      return ex.context();
    }

    friend constexpr boost::asio::execution::blocking_t::never_t query(
        const BThreadPoolExecutor&, boost::asio::execution::blocking_t) noexcept {
      return boost::asio::execution::blocking.never;
    }

    friend constexpr boost::asio::execution::outstanding_work_t::untracked_t query(
        const BThreadPoolExecutor&, boost::asio::execution::outstanding_work_t) noexcept {
      return boost::asio::execution::outstanding_work.untracked;
    }

    friend constexpr boost::asio::execution::relationship_t::fork_t query(
        const BThreadPoolExecutor&, boost::asio::execution::relationship_t) noexcept {
      return boost::asio::execution::relationship.fork;
    }

    friend BThreadPoolExecutor require(const BThreadPoolExecutor& ex,
                                       boost::asio::execution::blocking_t::never_t) noexcept {
      return ex;
    }

    friend BThreadPoolExecutor prefer(const BThreadPoolExecutor& ex,
                                      boost::asio::execution::blocking_t::possibly_t) noexcept {
      return ex;
    }

    friend BThreadPoolExecutor prefer(
        const BThreadPoolExecutor& ex,
        boost::asio::execution::outstanding_work_t::tracked_t) noexcept {
      return ex;
    }

    friend BThreadPoolExecutor prefer(
        const BThreadPoolExecutor& ex,
        boost::asio::execution::outstanding_work_t::untracked_t) noexcept {
      return ex;
    }

    friend BThreadPoolExecutor prefer(const BThreadPoolExecutor& ex,
                                      boost::asio::execution::relationship_t::fork_t) noexcept {
      return ex;
    }

    friend BThreadPoolExecutor prefer(
        const BThreadPoolExecutor& ex,
        boost::asio::execution::relationship_t::continuation_t) noexcept {
      return ex;
    }
#endif

    template <typename... Args>
    void post(Args&&... args) const {
      assert(pool_);
      pool_->post(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void defer(Args&&... args) const {
      assert(pool_);
      pool_->defer(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void dispatch(Args&&... args) const {
      assert(pool_);
      pool_->dispatch(std::forward<Args>(args)...);
    }

    context_type& context() const noexcept {
      assert(pool_);
#ifdef USE_BOOST_ASIO_EXECUTOR
      return static_cast<context_type&>(*pool_);
#else
      return *pool_;
#endif
    }

    friend bool operator==(const BThreadPoolExecutor& lhs,
                           const BThreadPoolExecutor& rhs) noexcept {
      return lhs.pool_ == rhs.pool_;
    }

    friend bool operator!=(const BThreadPoolExecutor& lhs,
                           const BThreadPoolExecutor& rhs) noexcept {
      return !(lhs == rhs);
    }

   private:
    BThreadPool* pool_;
    allocator_type allocator_;
  };

  using executor_type = BThreadPoolExecutor;

  BThreadPoolExecutor get_executor() noexcept {
    return BThreadPoolExecutor(this);
  }

  BThreadPoolExecutor get_executor() const noexcept {
    return BThreadPoolExecutor(const_cast<BThreadPool*>(this));
  }

  allocator_type get_allocator() const noexcept {
    return allocator_;
  }

 private:
  // Design note:
  // The pool deliberately uses one state mutex (`state_mtx_`) plus one
  // condition variable (`state_cv_`) for all ordering-sensitive operations:
  // submission, queue bookkeeping, worker sleep/wake, shrink, and lifecycle.
  //
  // The important sequencing rules are:
  // 1. Submission decides whether a new worker is needed before the task is
  //    accepted, so thread-start failures cannot leave a counted task behind
  //    with nobody able to execute it.
  // 2. A worker object is inserted into `workers_` before its std::thread is
  //    started, so join/shutdown can never miss a live worker.
  // 3. `join()` first flips the state to JOINING, waits for
  //    `pending_task_num_ == 0`, then moves workers out and joins them.
  // 4. `shutdown()` first flips the state to STOPPING, drops queued work under
  //    the same mutex, then wakes workers so running tasks can finish and exit.
  //
  // Keeping these steps under one lock removes the deadlock-prone "half old
  // state, half new state" windows that existed when semaphore permits,
  // pending counters, and worker ownership were updated independently.
  enum class QueuePreference : unsigned char { kPreferFast, kSlowOnly };
  enum Status : unsigned char { RUNNING, JOINING, STOPPING, STOPPED };

  class ThreadWorker {
   public:
    ThreadWorker() = default;
    ThreadWorker(const ThreadWorker&) = delete;
    ThreadWorker(ThreadWorker&&) noexcept = delete;
    ThreadWorker& operator=(const ThreadWorker&) = delete;
    ThreadWorker& operator=(ThreadWorker&&) noexcept = delete;

    ~ThreadWorker() {
      join();
    }

    void join() noexcept {
      if (thread_.joinable()) {
        if (thread_.get_id() == std::this_thread::get_id()) {
          thread_.detach();
        } else {
          thread_.join();
        }
      }
    }

    std::thread thread_;
    bool exited_{false};
  };

  bool should_accept_new_tasks_locked() const noexcept {
    return stat_ == RUNNING;
  }

  void on_task_accepted_locked() noexcept {
    ++pending_task_num_;
  }

  void on_task_finished_locked() noexcept {
    assert(pending_task_num_ > 0);
    --pending_task_num_;
  }

  size_t effective_max_thread_num() const noexcept {
    return std::max<std::size_t>(1, param_.max_thread_num);
  }

  size_t effective_core_thread_num() const noexcept {
    return std::min(effective_max_thread_num(),
                    std::max<std::size_t>(1, param_.core_thread_num));
  }

  bool use_fast_queue() const noexcept {
    return param_.fast_queue_capacity != 0;
  }

  bool has_queued_tasks_locked() const noexcept {
    return fast_queue_size_ != 0 || slow_queue_size_ != 0;
  }

  std::chrono::milliseconds idle_worker_timeout() const noexcept {
    return std::chrono::milliseconds(param_.thread_clean_interval);
  }

  bool should_start_worker_for_new_task_locked() const noexcept {
    if (live_worker_num_ < effective_core_thread_num()) {
      return true;
    }
    return idle_worker_num_ == 0 && live_worker_num_ < effective_max_thread_num();
  }

  template <typename Fn>
  ThreadFuncPtr make_thread_func(Fn&& fn) {
    ThreadFuncAllocator alloc(allocator_);
    auto* ptr = RebindAllocatorTraits<ThreadFunc>::allocate(alloc, 1);
    try {
      RebindAllocatorTraits<ThreadFunc>::construct(alloc, ptr, std::forward<Fn>(fn));
    } catch (...) {
      RebindAllocatorTraits<ThreadFunc>::deallocate(alloc, ptr, 1);
      throw;
    }
    return ptr;
  }

  void destroy_thread_func(ThreadFuncPtr ptr) noexcept {
    if (!ptr) {
      return;
    }
    ThreadFuncAllocator alloc(allocator_);
    RebindAllocatorTraits<ThreadFunc>::destroy(alloc, ptr);
    RebindAllocatorTraits<ThreadFunc>::deallocate(alloc, ptr, 1);
  }

  ThreadWorkerPtr make_thread_worker();

  ThreadFuncPtr pop_task_locked() {
    ThreadFuncPtr func = nullptr;
    if (fast_queue_size_ != 0 && fast_queue_.pop(func)) {
      --fast_queue_size_;
      return func;
    }
    if (slow_queue_size_ != 0 && slow_queue_.pop(func)) {
      --slow_queue_size_;
      return func;
    }
    return nullptr;
  }

  void clear_queued_tasks_locked() noexcept {
    ThreadFuncPtr func = nullptr;
    while (fast_queue_.pop(func)) {
      if (fast_queue_size_ != 0) {
        --fast_queue_size_;
      }
      destroy_thread_func(func);
      on_task_finished_locked();
    }
    while (slow_queue_.pop(func)) {
      if (slow_queue_size_ != 0) {
        --slow_queue_size_;
      }
      destroy_thread_func(func);
      on_task_finished_locked();
    }
    if (pending_task_num_ == 0) {
      state_cv_.notify_all();
    }
  }

  void mark_worker_exited_locked(ThreadWorker* worker) noexcept {
    assert(worker != nullptr);
    assert(live_worker_num_ > 0);
    // This flag is written while holding `state_mtx_`, and reaped later by
    // non-worker threads. That ordering matters: workers only decide "I am
    // done", while ownership transfer and `join()` happen elsewhere.
    worker->exited_ = true;
    --live_worker_num_;
    state_cv_.notify_all();
  }

  void start_worker_locked() {
    auto worker = make_thread_worker();
    auto* raw_worker = worker.get();
    workers_.emplace_back(std::move(worker));

    try {
      // Reason: register the worker object before the thread starts running.
      // Effect: `join()`/`shutdown()` always see every live worker and cannot
      // miss a thread that started between "spawn" and "bookkeeping".
      raw_worker->thread_ = std::thread([this, raw_worker] { worker_loop(raw_worker); });
      ++live_worker_num_;
    } catch (...) {
      auto it = std::find_if(workers_.begin(), workers_.end(),
                             [raw_worker](const ThreadWorkerPtr& current) {
                               return current.get() == raw_worker;
                             });
      if (it != workers_.end()) {
        workers_.erase(it);
      }
      throw;
    }
  }

  std::vector<ThreadWorkerPtr> collect_exited_workers() {
    std::vector<ThreadWorkerPtr> exited_workers;
    std::lock_guard<std::mutex> lock(state_mtx_);
    auto it = workers_.begin();
    while (it != workers_.end()) {
      if (*it && (*it)->exited_) {
        exited_workers.emplace_back(std::move(*it));
        it = workers_.erase(it);
      } else {
        ++it;
      }
    }
    return exited_workers;
  }

  void join_workers(std::vector<ThreadWorkerPtr>& workers) noexcept {
    for (auto& worker : workers) {
      if (worker) {
        // Join always happens outside `state_mtx_`, otherwise a worker trying to
        // report its final state would deadlock behind the joiner.
        worker->join();
      }
    }
  }

  void reap_exited_workers() {
    auto exited_workers = collect_exited_workers();
    join_workers(exited_workers);
  }

  bool enqueue_task(ThreadFuncPtr func_ptr, QueuePreference preference) {
    reap_exited_workers();

    std::lock_guard<std::mutex> lock(state_mtx_);
    if (!should_accept_new_tasks_locked()) {
      return false;
    }

    // Start or reserve execution capacity before the task becomes visible.
    // This ordering is intentional: if thread creation throws here, the caller
    // sees a failed submission instead of a permanently pending task.
    if (should_start_worker_for_new_task_locked()) {
      start_worker_locked();
    }

    on_task_accepted_locked();
    if (preference == QueuePreference::kPreferFast && use_fast_queue() && fast_queue_.push(func_ptr)) {
      ++fast_queue_size_;
    } else {
      slow_queue_.push(func_ptr);
      ++slow_queue_size_;
    }

    // Reason: workers now sleep on the same condition variable that protects
    // queue state and stop state.
    // Effect: one accepted task always translates into one wake-up signal, so
    // we avoid the lost-permit bugs from the previous semaphore-based design.
    state_cv_.notify_one();
    return true;
  }

  bool post(ThreadFuncPtr func_ptr) {
    try {
      return enqueue_task(func_ptr, QueuePreference::kPreferFast);
    } catch (...) {
      destroy_thread_func(func_ptr);
      throw;
    }
  }

  bool defer(ThreadFuncPtr func_ptr) {
    try {
      return enqueue_task(func_ptr, QueuePreference::kSlowOnly);
    } catch (...) {
      destroy_thread_func(func_ptr);
      throw;
    }
  }

  void worker_loop(ThreadWorker* worker) noexcept {
    struct WorkerScope {
      explicit WorkerScope(BThreadPool* pool) : previous_pool(BThreadPool::current_pool_) {
        BThreadPool::dispatch_depth_ = 1;
        BThreadPool::current_pool_ = pool;
      }
      ~WorkerScope() {
        BThreadPool::dispatch_depth_ = 0;
        BThreadPool::current_pool_ = previous_pool;
      }

      BThreadPool* previous_pool;
    } scope(this);

    for (;;) {
      ThreadFuncPtr func = nullptr;
      {
        std::unique_lock<std::mutex> lock(state_mtx_);
        for (;;) {
          // Queue selection is always done while holding `state_mtx_`. This is
          // slower than a pure lock-free fast path, but it gives join/shutdown
          // a single source of truth for "is there still visible work?".
          func = pop_task_locked();
          if (func) {
            break;
          }

          if (stat_ == STOPPING) {
            mark_worker_exited_locked(worker);
            return;
          }

          if (stat_ == JOINING) {
            // In JOINING we never accept new work again. Workers therefore only
            // have two legal outcomes: find already-queued work, or sleep until
            // the remaining in-flight work count reaches zero.
            if (pending_task_num_ == 0) {
              mark_worker_exited_locked(worker);
              return;
            }

            ++idle_worker_num_;
            state_cv_.wait(lock, [this] {
              return stat_ == STOPPING || has_queued_tasks_locked() || pending_task_num_ == 0;
            });
            --idle_worker_num_;
            continue;
          }

          ++idle_worker_num_;
          if (live_worker_num_ > effective_core_thread_num()) {
            // Reason: idle shrink is handled by a timed wait under the same
            // state mutex instead of a detached self-cleanup path.
            // Effect: extra threads retire without racing `join()`/`shutdown()`
            // for ownership of their worker objects.
            const bool woke_for_work =
                state_cv_.wait_for(lock, idle_worker_timeout(), [this] {
                  return stat_ != RUNNING || has_queued_tasks_locked();
                });
            --idle_worker_num_;

            if (!woke_for_work && stat_ == RUNNING && !has_queued_tasks_locked() &&
                live_worker_num_ > effective_core_thread_num()) {
              mark_worker_exited_locked(worker);
              return;
            }
          } else {
            // Core workers wait indefinitely. They are the minimum execution
            // capacity that guarantees forward progress even after temporary
            // bursts have drained away.
            state_cv_.wait(lock, [this] {
              return stat_ != RUNNING || has_queued_tasks_locked();
            });
            --idle_worker_num_;
          }
        }
      }

      try {
        (*func)();
      } catch (...) {
        // Ignore exceptions from fire-and-forget tasks.
      }
      destroy_thread_func(func);

      std::lock_guard<std::mutex> lock(state_mtx_);
      on_task_finished_locked();
      // Wake lifecycle waiters after every finish while stopping, and also wake
      // them when the last accepted task leaves the system.
      if (stat_ != RUNNING || pending_task_num_ == 0) {
        state_cv_.notify_all();
      }
    }
  }

  // Parameter of the thread pool.
  const BThreadPoolParam param_;
  allocator_type allocator_;

  // Task queues, including a fast queue and a slow queue.
  SlowQueueType slow_queue_;
  FastQueueType fast_queue_;

  std::mutex lifecycle_mtx_;
  std::mutex state_mtx_;
  std::condition_variable state_cv_;
  Status stat_;
  WorkerContainer workers_;
  std::size_t live_worker_num_;
  std::size_t idle_worker_num_;
  std::size_t pending_task_num_;
  std::size_t fast_queue_size_;
  std::size_t slow_queue_size_;

  // The max recursion depth limit.
  static constexpr std::size_t kMaxDispatchDepth = 32;

  // Use thread_local to make sure that the depth count is for every thread.
  static inline thread_local std::size_t dispatch_depth_ = 0;
  static inline thread_local BThreadPool* current_pool_;
};

template <typename Allocator>
inline void BThreadPool<Allocator>::ThreadWorkerDeleter::operator()(
    ThreadWorker* ptr) const noexcept {
  if (!ptr) {
    return;
  }
  RebindAllocator<ThreadWorker> alloc(allocator);
  RebindAllocatorTraits<ThreadWorker>::destroy(alloc, ptr);
  RebindAllocatorTraits<ThreadWorker>::deallocate(alloc, ptr, 1);
}

template <typename Allocator>
inline typename BThreadPool<Allocator>::ThreadWorkerPtr
BThreadPool<Allocator>::make_thread_worker() {
  RebindAllocator<ThreadWorker> alloc(allocator_);
  auto* ptr = RebindAllocatorTraits<ThreadWorker>::allocate(alloc, 1);
  try {
    RebindAllocatorTraits<ThreadWorker>::construct(alloc, ptr);
  } catch (...) {
    RebindAllocatorTraits<ThreadWorker>::deallocate(alloc, ptr, 1);
    throw;
  }
  return ThreadWorkerPtr(ptr, ThreadWorkerDeleter{allocator_});
}

}  // namespace bthpool::detail

namespace bthpool {
using detail::BThreadPool;
using detail::BThreadPoolParam;
}  // namespace bthpool

#endif
