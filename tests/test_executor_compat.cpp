#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>

#include <gtest/gtest.h>

#include "bthpool/bthpool.hpp"

#if BTHPOOL_HAS_BOOST_ASIO
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/post.hpp>

namespace {
template <typename Rep, typename Period>
bool JoinWithTimeout(bthpool::BThreadPool<>* pool,
                     std::chrono::duration<Rep, Period> timeout) {
  std::promise<void> done;
  auto future = done.get_future();

  std::thread joiner([pool, done = std::move(done)]() mutable {
    pool->join();
    done.set_value();
  });

  if (future.wait_for(timeout) != std::future_status::ready) {
    joiner.detach();
    return false;
  }

  joiner.join();
  return true;
}
}  // namespace

TEST(ExecutorCompat, SupportsAnyIoExecutorTypeErasure) {
  using namespace std::chrono_literals;

  static_assert(std::is_constructible_v<boost::asio::any_io_executor,
                                        bthpool::BThreadPool<>::executor_type>);

  auto pool = std::make_unique<bthpool::BThreadPool<>>();
  boost::asio::any_io_executor ex(pool->get_executor());

  std::promise<int> promise;
  auto future = promise.get_future();

  boost::asio::post(ex, [&promise] { promise.set_value(42); });
  ASSERT_EQ(future.wait_for(2s), std::future_status::ready)
      << "post did not complete within timeout";
  EXPECT_EQ(future.get(), 42);

  if (!JoinWithTimeout(pool.get(), 2s)) {
    pool.release();
    FAIL() << "join timed out";
  }
  pool.reset();
}

TEST(ExecutorCompat, RepeatedPoolLifecycleAndPostWakeup) {
  using namespace std::chrono_literals;

  constexpr std::size_t kIterations = 5000;
  constexpr auto kWaitTimeout = 50ms;
  constexpr auto kJoinTimeout = 2s;

  for (std::size_t i = 0; i < kIterations; ++i) {
    bthpool::BThreadPoolParam param;
    param.core_thread_num = 2;
    param.max_thread_num = 2;
    param.fast_queue_capacity = 16;

    auto pool = std::make_unique<bthpool::BThreadPool<>>(param);

    std::promise<void> done;
    auto future = done.get_future();
    boost::asio::post(pool->get_executor(), [&done] { done.set_value(); });

    ASSERT_EQ(future.wait_for(kWaitTimeout), std::future_status::ready)
        << "timeout at iteration " << i;

    if (!JoinWithTimeout(pool.get(), kJoinTimeout)) {
      pool.release();
      FAIL() << "join timeout at iteration " << i;
    }
    pool.reset();
  }
}
#else
TEST(ExecutorCompat, RequiresBoostAsioForAnyIoExecutor) {
  GTEST_SKIP() << "Boost.Asio support is disabled.";
}
#endif
