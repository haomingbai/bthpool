// Example: Multiple producer threads submitting work concurrently.
// Use this pattern when tasks are produced from several threads at once.

#include <bthpool/bthpool.hpp>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  bthpool::BThreadPoolParam param;
  param.core_thread_num = 4;
  param.max_thread_num = 8;
  param.fast_queue_capacity = 128;

  bthpool::BThreadPool pool(param);

  std::atomic<int> total{0};
  const int producers = 4;
  const int tasks_per_producer = 100;

  std::vector<std::thread> threads;
  threads.reserve(producers);
  for (int p = 0; p < producers; ++p) {
    threads.emplace_back([&] {
      for (int i = 0; i < tasks_per_producer; ++i) {
        pool.post([&total] { total.fetch_add(1, std::memory_order_relaxed); });
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  pool.join();

  std::cout << "total tasks executed: " << total.load() << std::endl;
  return 0;
}
