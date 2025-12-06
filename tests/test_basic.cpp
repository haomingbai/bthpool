#include <atomic>
#include <cassert>
#include <iostream>

#include "bthpool/bthpool.hpp"

int main() {
  bthpool::BThreadPool pool;

  std::atomic<int> sum{0};
  const int n = 10;
  for (int i = 1; i <= n; ++i) {
    pool.post([i, &sum] () mutable { sum.fetch_add(i, std::memory_order_relaxed); });
  }

  pool.join();

  int expected = (n * (n + 1)) / 2;  // 1..n sum formula
  std::cout << "computed=" << sum.load() << ", expected=" << expected << std::endl;
  assert(sum.load() == expected);
  return 0;
}
