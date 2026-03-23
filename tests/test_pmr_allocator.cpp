#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <type_traits>

#include "bthpool/bthpool.hpp"

namespace {

struct CountingAllocatorState {
  std::atomic<std::size_t> allocate_count{0};
  std::atomic<std::size_t> deallocate_count{0};
  std::atomic<std::ptrdiff_t> live_bytes{0};
};

template <typename T>
class CountingAllocator {
 public:
  using value_type = T;

  CountingAllocator() : state_(std::make_shared<CountingAllocatorState>()) {}

  explicit CountingAllocator(std::shared_ptr<CountingAllocatorState> state)
      : state_(std::move(state)) {
    if (!state_) {
      state_ = std::make_shared<CountingAllocatorState>();
    }
  }

  CountingAllocator(const CountingAllocator&) noexcept = default;
  CountingAllocator& operator=(const CountingAllocator&) noexcept = default;

  // Keep moved-from allocator valid by sharing state instead of emptying it.
  CountingAllocator(CountingAllocator&& other) noexcept : state_(other.state_) {}
  CountingAllocator& operator=(CountingAllocator&& other) noexcept {
    state_ = other.state_;
    return *this;
  }

  template <typename U>
  CountingAllocator(const CountingAllocator<U>& other) noexcept : state_(other.state_) {}

  T* allocate(std::size_t n) {
    state_->allocate_count.fetch_add(1, std::memory_order_relaxed);
    state_->live_bytes.fetch_add(static_cast<std::ptrdiff_t>(n * sizeof(T)),
                                 std::memory_order_relaxed);
    return std::allocator<T>{}.allocate(n);
  }

  void deallocate(T* ptr, std::size_t n) noexcept {
    if (!ptr) {
      return;
    }
    state_->deallocate_count.fetch_add(1, std::memory_order_relaxed);
    state_->live_bytes.fetch_sub(static_cast<std::ptrdiff_t>(n * sizeof(T)),
                                 std::memory_order_relaxed);
    std::allocator<T>{}.deallocate(ptr, n);
  }

  std::shared_ptr<CountingAllocatorState> state() const noexcept {
    return state_;
  }

  template <typename U>
  bool operator==(const CountingAllocator<U>& other) const noexcept {
    return state_ == other.state_;
  }

  template <typename U>
  bool operator!=(const CountingAllocator<U>& other) const noexcept {
    return !(*this == other);
  }

 private:
  template <typename>
  friend class CountingAllocator;

  std::shared_ptr<CountingAllocatorState> state_;
};

using PoolAllocator = CountingAllocator<std::byte>;
using Pool = bthpool::BThreadPool<PoolAllocator>;

}  // namespace

TEST(AllocatorMode, UsesProvidedAllocatorForTasks) {
  auto state = std::make_shared<CountingAllocatorState>();

  {
    bthpool::BThreadPoolParam param;
    Pool pool(param, PoolAllocator{state});

    std::promise<void> promise;
    auto future = promise.get_future();
    pool.post([&promise] { promise.set_value(); });
    future.get();
    pool.join();
  }

  EXPECT_GT(state->allocate_count.load(std::memory_order_relaxed), 0u);
  EXPECT_GT(state->deallocate_count.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(state->live_bytes.load(std::memory_order_relaxed), 0);
}

TEST(AllocatorMode, UsesDefaultAllocatorWhenNotProvided) {
  std::shared_ptr<CountingAllocatorState> state;

  {
    bthpool::BThreadPoolParam param;
    Pool pool(param);
    state = pool.get_allocator().state();

    std::promise<void> promise;
    auto future = promise.get_future();
    pool.post([&promise] { promise.set_value(); });
    future.get();
    pool.join();
  }

  ASSERT_NE(state, nullptr);
  EXPECT_GT(state->allocate_count.load(std::memory_order_relaxed), 0u);
  EXPECT_GT(state->deallocate_count.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(state->live_bytes.load(std::memory_order_relaxed), 0);
}

TEST(AllocatorMode, ExecutorAllocatorUsesPoolAllocator) {
  auto state = std::make_shared<CountingAllocatorState>();
  bthpool::BThreadPoolParam param;
  Pool pool(param, PoolAllocator{state});

  auto ex = pool.get_executor();
  auto alloc = ex.get_allocator();

  static_assert(std::is_same_v<decltype(alloc), PoolAllocator>);
  EXPECT_EQ(alloc.state(), state);

  pool.join();
}
