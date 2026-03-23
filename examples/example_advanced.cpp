// Example: Configure BThreadPool parameters and mix fast/slow queue submissions.
// Use this pattern when you need explicit control over sizing and scheduling behavior.

#include <bthpool/bthpool.hpp>
#include <future>
#include <iostream>
#include <vector>

int main() {
  bthpool::BThreadPoolParam param;
  param.core_thread_num = 2;
  param.max_thread_num = 6;
  param.fast_queue_capacity = 64;
  param.thread_clean_interval = 200;

  bthpool::BThreadPool<> pool(param);

  for (int i = 0; i < 8; ++i) {
    pool.post([i] { (void)i; });
  }

  pool.defer([] { /* lower-priority work */ });

  std::vector<std::future<int>> futures;
  for (int i = 0; i < 4; ++i) {
    futures.push_back(pool.futured_post([i] { return i * i; }));
  }

  pool.join();

  int sum = 0;
  for (auto& f : futures) {
    sum += f.get();
  }
  std::cout << "sum of squares: " << sum << std::endl;
  return 0;
}
