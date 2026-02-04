#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#include "test_common.hpp"

TEST(LockfreeQueueBasic, SafeQueueFIFOAndEmpty) {
  test_adapter::SafeQueue<int> q;
  int v = 0;
  EXPECT_FALSE(q.pop(v));
  q.push(1);
  q.push(2);
  q.push(3);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 1);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 2);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 3);
  EXPECT_FALSE(q.pop(v));
}

TEST(LockfreeQueueBasic, LockfreeFIFOAndCapacity) {
  test_adapter::LockfreeFixedQueue<uint64_t> q(4);
  uint64_t v = 0;
  EXPECT_TRUE(q.push(1));
  EXPECT_TRUE(q.push(2));
  EXPECT_TRUE(q.push(3));
  EXPECT_TRUE(q.push(4));
  EXPECT_FALSE(q.push(5)) << "queue should be full";

  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 1u);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 2u);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 3u);
  EXPECT_TRUE(q.pop(v));
  EXPECT_EQ(v, 4u);
  EXPECT_FALSE(q.pop(v));
}

TEST(LockfreeQueueBasic, MPMCNoLossNoDup) {
  const int producers = 4;
  const int consumers = 4;
  const uint64_t total = 50000;

  test_adapter::LockfreeFixedQueue<uint64_t> q(1024);
  std::atomic<uint64_t> next_id{0};
  std::atomic<uint64_t> consumed{0};

  std::vector<std::atomic<uint8_t>> seen(total);
  for (uint64_t i = 0; i < total; ++i) {
    seen[i].store(0, std::memory_order_relaxed);
  }
  test_util::DuplicateInfo dup;

  std::vector<std::thread> prod_threads;
  std::vector<std::thread> cons_threads;
  prod_threads.reserve(producers);
  cons_threads.reserve(consumers);

  for (int p = 0; p < producers; ++p) {
    prod_threads.emplace_back([&] {
      for (;;) {
        uint64_t id = next_id.fetch_add(1, std::memory_order_relaxed);
        if (id >= total) {
          break;
        }
        while (!q.push(id)) {
          std::this_thread::yield();
        }
      }
    });
  }

  for (int c = 0; c < consumers; ++c) {
    cons_threads.emplace_back([&] {
      uint64_t id = 0;
      while (consumed.load(std::memory_order_relaxed) < total) {
        if (q.pop(id)) {
          if (id < total) {
            uint8_t prev = seen[id].fetch_add(1, std::memory_order_relaxed);
            if (prev != 0) {
              test_util::record_duplicate(dup, id);
            }
          }
          consumed.fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& t : prod_threads) {
    t.join();
  }
  for (auto& t : cons_threads) {
    t.join();
  }

  EXPECT_EQ(consumed.load(std::memory_order_relaxed), total);
  EXPECT_EQ(dup.count.load(std::memory_order_relaxed), 0u)
      << "first duplicate id=" << dup.first_id.load(std::memory_order_relaxed);

  uint64_t missing = std::numeric_limits<uint64_t>::max();
  for (uint64_t i = 0; i < total; ++i) {
    if (seen[i].load(std::memory_order_relaxed) == 0) {
      missing = i;
      break;
    }
  }
  EXPECT_EQ(missing, std::numeric_limits<uint64_t>::max())
      << "missing id=" << missing;
}
