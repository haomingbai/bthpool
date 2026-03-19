#include <future>
#include <type_traits>

#include <gtest/gtest.h>

#include "bthpool/bthpool.hpp"

#if BTHPOOL_HAS_BOOST_ASIO
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/post.hpp>

TEST(ExecutorCompat, SupportsAnyIoExecutorTypeErasure) {
  static_assert(std::is_constructible_v<boost::asio::any_io_executor,
                                        bthpool::BThreadPool::executor_type>);

  bthpool::BThreadPool pool;
  boost::asio::any_io_executor ex(pool.get_executor());

  std::promise<int> promise;
  auto future = promise.get_future();

  boost::asio::post(ex, [&promise] { promise.set_value(42); });
  EXPECT_EQ(future.get(), 42);

  pool.join();
}
#else
TEST(ExecutorCompat, RequiresBoostAsioForAnyIoExecutor) {
  GTEST_SKIP() << "Boost.Asio support is disabled.";
}
#endif
