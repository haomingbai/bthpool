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

#pragma once
#ifndef BTHPOOL_BTHPOOL_HPP_
#define BTHPOOL_BTHPOOL_HPP_

#ifdef USE_BOOST_ASIO_EXECUTOR
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/execution.hpp>
#include <boost/asio/execution_context.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
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

  using ThreadFunc = std::move_only_function<void()>;
  using ThreadFuncPtr = ThreadFunc*;
  using ThreadFuncAllocator = RebindAllocator<ThreadFunc>;
  using SlowQueueContainer = std::deque<ThreadFuncPtr, RebindAllocator<ThreadFuncPtr>>;
  using SlowQueueType = SafeQueue<ThreadFuncPtr, SlowQueueContainer>;
  using FastQueueType = LockfreeFixedQueue<ThreadFuncPtr, RebindAllocator<ThreadFuncPtr>>;
  using ThreadMapValueType = std::pair<const std::thread::id, ThreadWorkerPtr>;
  using ThreadMapAllocator = RebindAllocator<ThreadMapValueType>;
  using ThreadMapType =
      std::unordered_map<std::thread::id, ThreadWorkerPtr, std::hash<std::thread::id>,
                         std::equal_to<std::thread::id>, ThreadMapAllocator>;

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
        task_counter_(0),
        thread_map_(0, std::hash<std::thread::id>{}, std::equal_to<std::thread::id>{},
                    ThreadMapAllocator(allocator_)),
        stat_(RUNNING),
        living_thread_num_(0),
        pending_task_num_(0) {}

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
    stat_.store(STOPPING, std::memory_order_release);

    {
      std::unique_lock<std::mutex> lock(pending_mtx_);
      pending_cv_.wait(lock, [this] {
        return pending_task_num_.load(std::memory_order_acquire) == 0;
      });
    }

    std::vector<ThreadWorkerPtr> workers;
    {
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
      std::lock_guard<std::mutex> lock(map_mtx_);
      workers.reserve(thread_map_.size());
      for (auto& [_, worker] : thread_map_) {
        if (worker) {
          workers.emplace_back(std::move(worker));
        }
      }
      thread_map_.clear();
    }

    for (auto& worker : workers) {
      if (worker) {
        worker->set_stop();
      }
    }

    if (!workers.empty()) {
      task_counter_.release(static_cast<std::ptrdiff_t>(workers.size()));
    }

    for (auto& worker : workers) {
      if (worker) {
        worker->join();
      }
    }

    living_thread_num_.store(0, std::memory_order_release);
    while (task_counter_.try_acquire()) {
    }
    stat_.store(STOPPED, std::memory_order_release);
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
    stat_.store(STOPPED, std::memory_order_release);
    // Clean the queues and release resources.
    ThreadFuncPtr func_ptr;
    while (fast_queue_.pop(func_ptr)) {
      destroy_thread_func(func_ptr);
      on_task_finished();
    }
    while (slow_queue_.pop(func_ptr)) {
      destroy_thread_func(func_ptr);
      on_task_finished();
    }

    std::vector<ThreadWorkerPtr> workers;
    {
      std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
      std::lock_guard<std::mutex> lock(map_mtx_);
      workers.reserve(thread_map_.size());
      for (auto& [_, worker] : thread_map_) {
        if (worker) {
          worker->set_stop();
          workers.emplace_back(std::move(worker));
        }
      }
      thread_map_.clear();
    }

    if (!workers.empty()) {
      task_counter_.release(static_cast<std::ptrdiff_t>(workers.size()));
    }

    for (auto& worker : workers) {
      if (worker) {
        worker->join();
      }
    }

    living_thread_num_.store(0, std::memory_order_release);
    while (task_counter_.try_acquire()) {
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
  bool should_accept_new_tasks() const noexcept {
    return stat_.load(std::memory_order_acquire) == RUNNING;
  }

  void on_task_accepted() noexcept {
    pending_task_num_.fetch_add(1, std::memory_order_acq_rel);
  }

  void on_task_finished() noexcept {
    auto prev = pending_task_num_.fetch_sub(1, std::memory_order_acq_rel);
    assert(prev > 0);
    if (prev == 1) {
      pending_cv_.notify_all();
    }
  }

  size_t effective_core_thread_num() const noexcept {
    // Keep one always-available worker even when core_thread_num is configured
    // as 0, so queued tasks can still make forward progress.
    return param_.core_thread_num == 0 ? 1 : param_.core_thread_num;
  }

  bool use_fast_queue() const noexcept {
    return param_.fast_queue_capacity != 0;
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

  bool post(ThreadFuncPtr func_ptr) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
    if (!should_accept_new_tasks()) {
      return false;
    }
    on_task_accepted();
    if (!should_accept_new_tasks()) {
      on_task_finished();
      return false;
    }

    bool queued = false;
    try {
    const auto effective_core = effective_core_thread_num();
    auto curr_num = living_thread_num_.load(std::memory_order_acquire);
    while (curr_num < effective_core) {
      if (living_thread_num_.compare_exchange_weak(
              curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Get the lock.
        // Create a thread to execute the task immediately.
        auto worker_ptr = make_thread_worker();
        ThreadWorker::run(this, std::move(worker_ptr));
        break;
      }
    }
    // Push the task to the queue.
    if (use_fast_queue() && fast_queue_.push(func_ptr)) {
      queued = true;
      task_counter_.release();
      return true;
    } else {
      curr_num = living_thread_num_.load(std::memory_order_acquire);
      // Create a new thread when all threads are occupied and new threads
      // are available.
      while (curr_num < param_.max_thread_num) {
        if (living_thread_num_.compare_exchange_weak(
                curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          // Get the lock.
          // Create a thread to execute the task immediately.
          auto worker_ptr = make_thread_worker();
          ThreadWorker::run(this, std::move(worker_ptr));
          break;
        }
      }
      // Push the task into the slow queue to wait.
      slow_queue_.push(func_ptr);
      queued = true;
      task_counter_.release();
      return true;
    }
    } catch (...) {
      if (!queued) {
        destroy_thread_func(func_ptr);
      }
      on_task_finished();
      throw;
    }
  }

  bool defer(ThreadFuncPtr func_ptr) {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mtx_);
    if (!should_accept_new_tasks()) {
      return false;
    }
    on_task_accepted();
    if (!should_accept_new_tasks()) {
      on_task_finished();
      return false;
    }

    bool queued = false;
    try {
    const auto effective_core = effective_core_thread_num();
    // If the thread is less than expected, then create a new one.
    auto curr_num = living_thread_num_.load(std::memory_order_acquire);
    while (curr_num < effective_core) {
      if (living_thread_num_.compare_exchange_weak(
              curr_num, curr_num + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Get the lock.
        // Create a thread to execute the task immediately.
        auto worker_ptr = make_thread_worker();
        ThreadWorker::run(this, std::move(worker_ptr));
        break;
      }
    }
    // Push the task into the slow queue to wait directly.
    slow_queue_.push(func_ptr);
    queued = true;
    task_counter_.release();
    return true;
    } catch (...) {
      if (!queued) {
        destroy_thread_func(func_ptr);
      }
      on_task_finished();
      throw;
    }
  }

  class ThreadWorker {
   public:
    static void run(BThreadPool* const pool, ThreadWorkerPtr self) noexcept;

    ThreadWorker() : should_stop_(false) {}

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

    void detach() noexcept {
      std::lock_guard<std::mutex> lock(mtx_);
      if (thread_.joinable()) {
        thread_.detach();
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
    std::atomic<bool> should_stop_;
  };

  class ThreadWorkerFunctor {
   public:
    explicit ThreadWorkerFunctor(BThreadPool* const pool, ThreadWorker* worker)
        : pool_(pool), worker_(worker) {}

    void operator()() noexcept {
      // Set some runtime status, to mark that
      // the thread running is in the thread pool.
      dispatch_depth_++;
      pool_->current_pool_ = pool_;

      for (;;) {
        auto status = pool_->stat_.load(std::memory_order_acquire);
        if (status == STOPPING) {
          auto func = try_get_task();
          if (func) {
            execute_and_delete_function(func);
            continue;
          }
          if (worker_->should_stop() ||
              pool_->pending_task_num_.load(std::memory_order_acquire) == 0) {
            break;
          }
          std::this_thread::yield();
          continue;
        }

        if (!pool_->task_counter_.try_acquire()) {
          if (try_cleanup()) {
            return;
          }
          pool_->task_counter_.acquire();
        }

        auto func = try_get_task();
        if (!func) {
          auto current_status = pool_->stat_.load(std::memory_order_acquire);
          if (current_status == STOPPED || worker_->should_stop()) {
            break;
          }
          std::this_thread::yield();
          continue;
        }
        execute_and_delete_function(func);
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
        pool_->destroy_thread_func(func);
        pool_->on_task_finished();
      }
    }

    ThreadFuncPtr try_get_task() {
      ThreadFuncPtr func = nullptr;
      // Try to first
      auto succ = pool_->fast_queue_.pop(func);
      if (succ) {
        return func;
      } else if ((succ = pool_->slow_queue_.pop(func))) {
        return func;
      }
      return nullptr;
    }

    bool try_cleanup() {
      if (pool_->stat_.load(std::memory_order_acquire) != RUNNING) {
        return false;
      }
      std::unique_lock<std::mutex> lifecycle_lock(pool_->lifecycle_mtx_);
      if (pool_->stat_.load(std::memory_order_acquire) != RUNNING) {
        return false;
      }
      if (pool_->pending_task_num_.load(std::memory_order_acquire) > 0) {
        return false;
      }
      const auto effective_core = pool_->effective_core_thread_num();
      std::ptrdiff_t curr_num = pool_->living_thread_num_.load(std::memory_order_acquire);
      while (curr_num > effective_core) {
        if (pool_->living_thread_num_.compare_exchange_weak(
                curr_num, curr_num - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          // Successfully get the lock.
          std::unique_lock<std::mutex> lock(pool_->map_mtx_);
          auto tid = std::this_thread::get_id();
          auto it = pool_->thread_map_.find(tid);
          if (it != pool_->thread_map_.end()) {
            // Move ownership out of the worker map for async cleanup task.
            auto worker_ptr(std::move(it->second));
            pool_->thread_map_.erase(it);
            // Release the lock to avoid dead lock before posting.
            lock.unlock();
            lifecycle_lock.unlock();
            // Cleanup in a move-only task.
            worker_ptr->set_stop();
            worker_ptr->detach();
            worker_ptr.reset();
            return true;
          } else {
            pool_->living_thread_num_.fetch_add(1, std::memory_order_acq_rel);
            return false;
          }
        }
      }
      // The number of thread is same as or lower than the core thread number.
      // No need to clean.
      return false;
    }

    ThreadWorker* const worker_;
    // Temperory sotre the pointer of the thread pool.
    BThreadPool* const pool_;
  };

  // Parameter of the thread pool.
  const BThreadPoolParam param_;
  allocator_type allocator_;

  // Task queues, including a fast queue and a slow queue.
  SlowQueueType slow_queue_;
  FastQueueType fast_queue_;
  std::counting_semaphore<> task_counter_;

  std::mutex lifecycle_mtx_;
  std::mutex pending_mtx_;
  std::condition_variable pending_cv_;

  // Thread map, which can find the thread worker and clean.
  std::mutex map_mtx_;
  ThreadMapType thread_map_;

  // Determine whether the pool should stop.
  enum Status : unsigned char { RUNNING, STOPPING, STOPPED };
  std::atomic<Status> stat_;

  // Indicate the number of working thread.
  std::atomic<std::ptrdiff_t> living_thread_num_;
  std::atomic<std::ptrdiff_t> pending_task_num_;

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

template <typename Allocator>
inline void BThreadPool<Allocator>::ThreadWorker::run(BThreadPool* const pool,
                                                      ThreadWorkerPtr self) noexcept {
  self->func_ = ThreadWorkerFunctor{pool, self.get()};
  self->thread_ = std::thread(std::move(self->func_));
  auto tid = self->thread_.get_id();
  {
    std::lock_guard<std::mutex> lock(pool->map_mtx_);
    pool->thread_map_.emplace(tid, std::move(self));
  }
}

}  // namespace bthpool::detail

namespace bthpool {
using detail::BThreadPool;
using detail::BThreadPoolParam;
}  // namespace bthpool

#endif
