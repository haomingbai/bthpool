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

  // Wait and quit
  pool.join();
  std::cout << "sum of squares: " << sum.load() << std::endl;

  return 0;
}
