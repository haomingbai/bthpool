#include <atomic>
#include <cassert>
#include <iostream>
#include <vector>

#include "bthpool/bthpool.hpp"

int main() {
  bthpool::BThreadPool pool;

  std::atomic<int> sum{0};
  const int n = 10;
  for (int i = 1; i <= n; ++i) {
    pool.post([i, &sum] () mutable { sum.fetch_add(i, std::memory_order_relaxed); });
  }

  // Validate futured_post for non-void
  auto f1 = pool.futured_post([] { return 7; });
  auto f2 = pool.futured_post([](int x) { return x * 2; }, 9);

  // Validate futured_post for void
  auto f3 = pool.futured_post([&sum] { sum.fetch_add(5, std::memory_order_relaxed); });

  pool.join();

  int expected = (n * (n + 1)) / 2 + 5;  // 1..n sum formula and the 5 by f3
  std::cout << "computed=" << sum.load() << ", expected=" << expected << std::endl;
  assert(sum.load() == expected);

  // Check futures
  assert(f1.get() == 7);
  assert(f2.get() == 18);
  f3.get();  // should not throw
  return 0;
}
