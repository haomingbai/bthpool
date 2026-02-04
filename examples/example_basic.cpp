// Example: Minimal BThreadPool usage with a basic task and a future result.
// Use this pattern when you want a small, straightforward pool with graceful shutdown.

#include <bthpool/bthpool.hpp>
#include <iostream>

int main() {
  bthpool::BThreadPool pool;

  pool.post([] { std::cout << "hello from worker" << std::endl; });

  auto fut = pool.futured_post([](int a, int b) { return a + b; }, 2, 40);

  pool.join();

  std::cout << "sum: " << fut.get() << std::endl;
  return 0;
}
