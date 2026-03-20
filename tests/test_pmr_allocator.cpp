#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <future>
#include <memory_resource>
#include <type_traits>

#include "bthpool/bthpool.hpp"

namespace {

class CountingResource : public std::pmr::memory_resource {
 public:
  explicit CountingResource(std::pmr::memory_resource* upstream = std::pmr::new_delete_resource())
      : upstream_(upstream) {}

  std::size_t allocate_count() const noexcept {
    return allocate_count_.load(std::memory_order_relaxed);
  }

  std::size_t deallocate_count() const noexcept {
    return deallocate_count_.load(std::memory_order_relaxed);
  }

  std::ptrdiff_t live_bytes() const noexcept {
    return live_bytes_.load(std::memory_order_relaxed);
  }

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    allocate_count_.fetch_add(1, std::memory_order_relaxed);
    live_bytes_.fetch_add(static_cast<std::ptrdiff_t>(bytes), std::memory_order_relaxed);
    return upstream_->allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    deallocate_count_.fetch_add(1, std::memory_order_relaxed);
    live_bytes_.fetch_sub(static_cast<std::ptrdiff_t>(bytes), std::memory_order_relaxed);
    upstream_->deallocate(p, bytes, alignment);
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

 private:
  std::pmr::memory_resource* upstream_;
  std::atomic<std::size_t> allocate_count_{0};
  std::atomic<std::size_t> deallocate_count_{0};
  std::atomic<std::ptrdiff_t> live_bytes_{0};
};

class ScopedDefaultResource {
 public:
  explicit ScopedDefaultResource(std::pmr::memory_resource* resource)
      : old_(std::pmr::set_default_resource(resource)) {}

  ~ScopedDefaultResource() {
    std::pmr::set_default_resource(old_);
  }

  ScopedDefaultResource(const ScopedDefaultResource&) = delete;
  ScopedDefaultResource& operator=(const ScopedDefaultResource&) = delete;

 private:
  std::pmr::memory_resource* old_;
};

}  // namespace

TEST(PmrAllocator, UsesProvidedResourceForTasks) {
  CountingResource resource;
  {
    bthpool::BThreadPoolParam param;
    param.memory_resource = &resource;

    bthpool::BThreadPool pool(param);
    std::promise<void> promise;
    auto future = promise.get_future();
    pool.post([&promise] { promise.set_value(); });
    future.get();
    pool.join();
  }

  EXPECT_GT(resource.allocate_count(), 0u);
  EXPECT_GT(resource.deallocate_count(), 0u);
  EXPECT_EQ(resource.live_bytes(), 0);
}

TEST(PmrAllocator, FallsBackToDefaultResourceWhenNull) {
  CountingResource default_resource;
  ScopedDefaultResource guard(&default_resource);

  {
    bthpool::BThreadPoolParam param;
    param.memory_resource = nullptr;

    bthpool::BThreadPool pool(param);
    std::promise<void> promise;
    auto future = promise.get_future();
    pool.post([&promise] { promise.set_value(); });
    future.get();
    pool.join();
  }

  EXPECT_GT(default_resource.allocate_count(), 0u);
  EXPECT_GT(default_resource.deallocate_count(), 0u);
  EXPECT_EQ(default_resource.live_bytes(), 0);
}

TEST(PmrAllocator, ExecutorAllocatorUsesPoolResource) {
  CountingResource resource;
  bthpool::BThreadPoolParam param;
  param.memory_resource = &resource;

  bthpool::BThreadPool pool(param);
  auto ex = pool.get_executor();
  auto alloc = ex.get_allocator();

  static_assert(std::is_same_v<decltype(alloc), std::pmr::polymorphic_allocator<void>>);
  EXPECT_EQ(alloc.resource(), &resource);

  pool.join();
}
