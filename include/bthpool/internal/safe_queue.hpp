/**
 * @file safe_queue.hpp
 * @brief Thread-safe queues for task scheduling.
 *
 * Provides two queue implementations used by the thread pool:
 * - `SafeQueue<T>`: mutex-based, simple FIFO for general usage.
 * - `LockfreeFixedQueue<T>`: lock-free, fixed-capacity ring buffer optimized
 *   for high-throughput scenarios.
 *
 * Both are designed to be used internally by `bthpool` to manage fast/slow
 * task paths. `LockfreeFixedQueue` requires `T` to be trivially destructible
 * and nothrow-constructible to guarantee non-blocking operations.
 *
 * Usage Sketch:
 * @code
 *   SafeQueue<int> q;
 *   q.push(1);
 *   int x;
 *   if (q.pop(x)) {  consume ; }
 *
 *   LockfreeFixedQueue<int> lq(1024);
 *   lq.push(42);
 *   int y;
 *   lq.pop(y);
 * @endcode
 *
 * @author  Haoming Bai <haomingbai@hotmail.com>
 * @date    2025-12-07
 * @version 0.1.0
 * @copyright Copyright © 2025 Haoming Bai
 * @license  MIT
 */

#pragma once

#ifndef BTHPOOL_INTERNAL_SAFE_QUEUE_HPP_
#define BTHPOOL_INTERNAL_SAFE_QUEUE_HPP_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>
#include <queue>
#include <type_traits>
#include <utility>

namespace bthpool {

inline namespace internal {

template <typename T, typename Container = std::deque<T, std::allocator<T>>>
class SafeQueue {
 public:
  SafeQueue(const SafeQueue&) = delete;
  SafeQueue(SafeQueue&&) noexcept = delete;
  SafeQueue& operator=(const SafeQueue&) = delete;
  SafeQueue& operator=(SafeQueue&&) noexcept = delete;
  ~SafeQueue() = default;

  SafeQueue() = default;

  void push(const T& elem) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push(elem);
  }

  void push(T&& elem) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push(std::move(elem));
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.emplace(std::forward<Args>(args)...);
  }

  bool pop(T& elem) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) {
      return false;
    } else {
      elem = std::move(queue_.front());
      queue_.pop();
      return true;
    }
  }

  template <typename E>
  bool pop(E& elem) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) {
      return false;
    } else {
      elem = std::move(queue_.front());
      queue_.pop();
      return true;
    }
  }

  bool pop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) {
      return false;
    } else {
      queue_.pop();
      return true;
    }
  }

  std::size_t empty() noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.empty();
  }

  std::size_t size() noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
  }

 private:
  std::mutex mtx_;
  std::queue<T, Container> queue_;
};

template <typename T>
class LockfreeFixedQueue {
  static_assert(std::is_trivially_destructible_v<T> && std::is_nothrow_constructible_v<T>,
                "LockfreeFixedQueue requires T to be trivially destructible "
                "and constructible without exceprion.");

 public:
  explicit LockfreeFixedQueue(std::size_t capacity)
      : head_(0),
        tail_(0),
        // The capacity of the queue should be the power of 2.
        capacity_(generate_actual_capacity(capacity)),
        // mask can help the program to get the actual index effieiently.
        mask_(capacity_ - 1),
        data_(new QueueNode[capacity_]) {
    for (std::size_t i = 0; i < capacity_; i++) {
      std::atomic_store_explicit(&data_[i].head_, std::numeric_limits<std::size_t>::max(),
                                 std::memory_order_release);
      std::atomic_store_explicit(&data_[i].tail_, i, std::memory_order_release);
    }
  }

  LockfreeFixedQueue() : LockfreeFixedQueue(kFallbackCapacity) {}

  LockfreeFixedQueue(const LockfreeFixedQueue&) = delete;
  LockfreeFixedQueue(LockfreeFixedQueue&&) noexcept = delete;
  LockfreeFixedQueue& operator=(const LockfreeFixedQueue&) = delete;
  LockfreeFixedQueue& operator=(LockfreeFixedQueue&&) noexcept = delete;

  ~LockfreeFixedQueue() {
    delete[] data_;
  }

  bool push(const T& elem) noexcept {
    // Store the node to be inserted.
    QueueNode* node = nullptr;
    // Get the tail to be inserted.
    auto tail = std::atomic_load_explicit(&tail_, std::memory_order_acquire);
    // Define whether the tail has been occupied.
    bool succ = false;
    for (std::size_t i = 0; i < kMaxSpinTime; i++) {
      // The node to be acquired.
      node = &data_[tail & mask_];
      // If the node's tail is not the expected tail,
      // which indicates that the current node has not been consumed.
      // This means that the queue is full, and the operation should fail.
      if (std::atomic_load_explicit(&node->tail_, std::memory_order_acquire) != tail) {
        return false;
      }
      // Try to acquire the node.
      if (std::atomic_compare_exchange_weak_explicit(
              &tail_, &tail, tail + 1,
              std::memory_order_acq_rel,  // success
              std::memory_order_relaxed   // failure (expected updated)
              )) {
        succ = true;
        break;
      }
    }
    if (succ) {
      node->construct_data(elem);
      // Store the new head of the node,
      // which will be checked while the the node is consumed.
      std::atomic_store_explicit(&node->head_, tail, std::memory_order_release);
      return true;
    } else {
      return false;
    }
  }

  bool push(T&& elem) noexcept {
    // Store the node to be inserted.
    QueueNode* node = nullptr;
    // Get the tail to be inserted.
    auto tail = std::atomic_load_explicit(&tail_, std::memory_order_acquire);
    // Define whether the tail has been occupied.
    bool succ = false;
    for (std::size_t i = 0; i < kMaxSpinTime; i++) {
      // The node to be acquired.
      node = &data_[tail & mask_];
      // If the node's tail is not the expected tail,
      // which indicates that the current node has not been consumed.
      // This means that the queue is full, and the operation should fail.
      if (std::atomic_load_explicit(&node->tail_, std::memory_order_acquire) != tail) {
        return false;
      }
      // Try to acquire the node.
      if (std::atomic_compare_exchange_weak_explicit(
              &tail_, &tail, tail + 1,
              std::memory_order_acq_rel,  // success
              std::memory_order_relaxed   // failure (expected updated)
              )) {
        succ = true;
        break;
      }
    }
    if (succ) {
      node->construct_data(std::move(elem));
      // Store the new head of the node,
      // which will be checked while the the node is consumed.
      std::atomic_store_explicit(&node->head_, tail, std::memory_order_release);
      return true;
    } else {
      return false;
    }
  }

  template <typename... Args>
  bool emplace(Args&&... args) noexcept {
    // Store the node to be inserted.
    QueueNode* node = nullptr;
    // Get the tail to be inserted.
    auto tail = std::atomic_load_explicit(&tail_, std::memory_order_acquire);
    // Define whether the tail has been occupied.
    bool succ = false;
    for (std::size_t i = 0; i < kMaxSpinTime; i++) {
      // The node to be acquired.
      node = &data_[tail & mask_];
      // If the node's tail is not the expected tail,
      // which indicates that the current node has not been consumed.
      // This means that the queue is full, and the operation should fail.
      if (std::atomic_load_explicit(&node->tail_, std::memory_order_acquire) != tail) {
        return false;
      }
      // Try to acquire the node.
      if (std::atomic_compare_exchange_weak_explicit(
              &tail_, &tail, tail + 1,
              std::memory_order_acq_rel,  // success
              std::memory_order_relaxed   // failure (expected updated)
              )) {
        succ = true;
        break;
      }
    }
    if (succ) {
      node->construct_data(std::forward<Args>(args)...);
      // Store the new head of the node,
      // which will be checked while the the node is consumed.
      std::atomic_store_explicit(&node->head_, tail, std::memory_order_release);
      return true;
    } else {
      return false;
    }
  }

  bool pop(T& elem) noexcept {
    // The pointer of the node to be poped.
    QueueNode* node = nullptr;
    // Get the head of the queue.
    auto head = std::atomic_load_explicit(&head_, std::memory_order_acquire);
    // Define whether the spin operation succeed.
    bool succ = false;
    for (std::size_t i = 0; i < kMaxSpinTime; i++) {
      // The node to be acquired.
      node = &data_[head & mask_];
      // If the node's head is not the expected tail,
      // which means that the queue is empty.
      if (std::atomic_load_explicit(&node->head_, std::memory_order_acquire) != head) {
        return false;
      }
      // The node is valid, try to occupy the node.
      if (std::atomic_compare_exchange_weak_explicit(
              &head_, &head, head + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        succ = true;
        break;
      }
    }
    if (succ) {
      elem = std::move(*(node->data_ptr()));
      std::atomic_store_explicit(&node->tail_, head + capacity_, std::memory_order_release);
      return true;
    } else {
      return false;
    }
  }

  size_t size() const noexcept {
    size_t head = std::atomic_load_explicit(&head_, std::memory_order_acquire);
    size_t tail = std::atomic_load_explicit(&tail_, std::memory_order_acquire);
    size_t diff = tail - head;  // unsigned arithmetic
    return (diff > capacity_) ? capacity_ : diff;
  }

  size_t capacity() const noexcept {
    return capacity_;
  }

 private:
  // The minimal capacity and the fallback capacity of the queue.
  static constexpr std::size_t kMinCapacity = 4;
  static constexpr std::size_t kFallbackCapacity = 1 << 10;
  static constexpr std::size_t kMaxCapacity =
      static_cast<size_t>(std::numeric_limits<unsigned int>::max()) + 1;

  // The maxium try number of the queue.
  static constexpr std::size_t kMaxSpinTime = 1 << 10;

  static inline std::size_t generate_actual_capacity(std::size_t capacity) {
    // The queue should have a capacity of at least 2.
    if (capacity <= 2) {
      return kMinCapacity;
    }
    // The capacity should not be to large.
    if (capacity > kMaxCapacity) {
      capacity = kMaxCapacity;
    }
    // If capacity has already been the power of 2, then return.
    if ((capacity & (capacity - 1)) == 0) {
      return capacity;
    }
    std::size_t actual_capacity = 1;
    while (actual_capacity < capacity) {
      assert(actual_capacity);
      actual_capacity <<= 1;
    }
    return actual_capacity;
  }

  struct QueueNode {
    alignas(alignof(T)) unsigned char data_[sizeof(T)];
    std::atomic<std::size_t> head_, tail_;

    T* data_ptr() noexcept {
      return std::launder(reinterpret_cast<T*>(data_));
    }

    const T* data_ptr() const noexcept {
      return std::launder(reinterpret_cast<const T*>(data_));
    }

    // Construct T in place (copy)
    void construct_data(const T& v) noexcept(std::is_nothrow_copy_constructible_v<T>) {
      new (data_) T(v);
    }

    // For completeness, also provide move construct (if desired)
    void construct_data(T&& v) noexcept(std::is_nothrow_move_constructible_v<T>) {
      new (data_) T(std::move(v));
    }

    // For other constructors.
    template <typename... Args>
    void construct_data(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
      static_assert(std::is_nothrow_constructible_v<T, Args&&...>,
                    "The constructor should not throw.");
      new (data_) T(std::forward<Args>(args)...);
    }
  };

  std::atomic<std::size_t> head_, tail_;
  const std::size_t capacity_;
  const std::size_t mask_;
  QueueNode* const data_;
};

}  // namespace internal

}  // namespace bthpool

#endif
