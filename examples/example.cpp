#include <atomic>
#include <iostream>
#include <vector>

#include "bthpool/bthpool.hpp"

int main() {
  std::cout << "bthpool example running..." << std::endl;

  bthpool::BThreadPool pool;

  std::atomic<int> sum{0};
  for (int i = 0; i < 8; ++i) {
    pool.post([i, &sum] { sum.fetch_add(i * i, std::memory_order_relaxed); });
  }

  // Use futured_post to get a result from a task
  auto fut_int = pool.futured_post([] { return 42; });
  // Use futured_post with arguments
  auto fut_add = pool.futured_post([](int a, int b) { return a + b; }, 10, 32);
  // futured_post with void return (completes without value)
  auto fut_void = pool.futured_post([&sum] { sum.fetch_add(1, std::memory_order_relaxed); });

  // Wait and quit
  pool.join();
  std::cout << "sum of squares: " << sum.load() << std::endl;
  std::cout << "future<int> results: " << fut_int.get() << ", " << fut_add.get() << std::endl;
  fut_void.get();

  return 0;
}
