#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "test_common.hpp"

TEST(ThreadPoolBasic, ConstructDestructNoHang) {
  auto pool = test_adapter::make_pool(1);
  std::atomic<int> sum{0};
  for (int i = 0; i < 10; ++i) {
    pool.post([&sum] { sum.fetch_add(1, std::memory_order_relaxed); });
  }
  pool.join();
  EXPECT_EQ(sum.load(std::memory_order_relaxed), 10);
}

TEST(ThreadPoolBasic, MultiThreadAllTasksOnce) {
  const int tasks = 5000;
  auto pool = test_adapter::make_pool(4);

  std::vector<std::atomic<int>> counts(tasks);
  for (int i = 0; i < tasks; ++i) {
    counts[i].store(0, std::memory_order_relaxed);
  }

  for (int i = 0; i < tasks; ++i) {
    pool.post(
        [i, &counts] { counts[i].fetch_add(1, std::memory_order_relaxed); });
  }

  pool.join();
  for (int i = 0; i < tasks; ++i) {
    EXPECT_EQ(counts[i].load(std::memory_order_relaxed), 1) << "task id=" << i;
  }
}

TEST(ThreadPoolBasic, FutureAndExceptionPropagation) {
  auto pool = test_adapter::make_pool(2);

  if constexpr (test_adapter::has_futured_post<test_adapter::Pool,
                                               int (*)()>()) {
    auto f1 = pool.futured_post([] { return 7; });
    auto f2 =
        pool.futured_post([] -> int { throw std::runtime_error("boom"); });
    auto f3 = pool.futured_post([] {});

    EXPECT_EQ(f1.get(), 7);
    EXPECT_THROW({ (void)f2.get(); }, std::runtime_error);
    EXPECT_NO_THROW(f3.get());
  }

  pool.join();
}

TEST(ThreadPoolBasic, JoinWaitsForTasks) {
  auto pool = test_adapter::make_pool(2);
  std::atomic<int> done{0};

  for (int i = 0; i < 20; ++i) {
    pool.post([&done] {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      done.fetch_add(1, std::memory_order_relaxed);
    });
  }

  pool.join();
  EXPECT_EQ(done.load(std::memory_order_relaxed), 20);
}

TEST(ThreadPoolBasic, ShutdownDoesNotDeadlock) {
  auto pool = test_adapter::make_pool(2);
  for (int i = 0; i < 50; ++i) {
    pool.post(
        [] { std::this_thread::sleep_for(std::chrono::milliseconds(1)); });
  }

  auto fut = std::async(std::launch::async, [&pool] { pool.shutdown(); });
  auto status = fut.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(status, std::future_status::ready) << "shutdown timeout";
}

TEST(ThreadPoolBasic, ConcurrentSubmitStability) {
  const int submit_threads = 6;
  const int per_thread_tasks = 2000;
  auto pool = test_adapter::make_pool(4);

  std::atomic<int> done{0};
  std::barrier start(submit_threads + 1);
  std::vector<std::thread> threads;
  threads.reserve(submit_threads);

  for (int t = 0; t < submit_threads; ++t) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      for (int i = 0; i < per_thread_tasks; ++i) {
        pool.post([&done] { done.fetch_add(1, std::memory_order_relaxed); });
      }
    });
  }

  start.arrive_and_wait();
  for (auto& th : threads) {
    th.join();
  }

  pool.join();
  EXPECT_EQ(done.load(std::memory_order_relaxed),
            submit_threads * per_thread_tasks);
}

TEST(ThreadPoolBasic, FastQueueCapacityZeroStillExecutesTasks) {
  bthpool::BThreadPoolParam param;
  param.core_thread_num = 2;
  param.max_thread_num = 2;
  param.fast_queue_capacity = 0;
  bthpool::BThreadPool<> pool(param);

  constexpr int kTaskCount = 200;
  std::atomic<int> done{0};
  for (int i = 0; i < kTaskCount; ++i) {
    pool.post([&done] { done.fetch_add(1, std::memory_order_relaxed); });
  }

  pool.join();
  EXPECT_EQ(done.load(std::memory_order_relaxed), kTaskCount);
}

TEST(ThreadPoolBasic, CoreZeroAndFastQueueZeroStillMakesProgress) {
  bthpool::BThreadPoolParam param;
  param.core_thread_num = 0;
  param.max_thread_num = 1;
  param.fast_queue_capacity = 0;
  bthpool::BThreadPool<> pool(param);

  constexpr int kTaskCount = 4;
  std::atomic<int> done{0};
  for (int i = 0; i < kTaskCount; ++i) {
    pool.post([&done] { done.fetch_add(1, std::memory_order_relaxed); });
  }

  pool.join();
  EXPECT_EQ(done.load(std::memory_order_relaxed), kTaskCount);
}
