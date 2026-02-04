// Example: Capture task exceptions with futured_post.
// Use this pattern when you need reliable error reporting from background tasks.

#include <bthpool/bthpool.hpp>
#include <iostream>
#include <stdexcept>

int main() {
  bthpool::BThreadPool pool;

  auto fut = pool.futured_post([]() -> int {
    throw std::runtime_error("task failed");
  });

  pool.join();

  try {
    (void)fut.get();
  } catch (const std::exception& ex) {
    std::cout << "caught exception: " << ex.what() << std::endl;
  }

  return 0;
}
