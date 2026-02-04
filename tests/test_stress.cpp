#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "test_common.hpp"

TEST(Stress, ThreadPoolQueueIntegration) {
  test_util::StressConfig cfg = test_util::get_stress_config();
  SCOPED_TRACE(::testing::Message()
               << "threads=" << cfg.threads << " ops=" << cfg.ops
               << " secs=" << cfg.secs << " seed=" << cfg.seed);

  const uint64_t total_ops = static_cast<uint64_t>(cfg.ops);
  const int producers = cfg.threads;
  const int consumers = cfg.threads;

  test_adapter::LockfreeFixedQueue<uint64_t> q(1 << 12);
  auto pool = test_adapter::make_pool(static_cast<std::size_t>(consumers));

  std::vector<std::atomic<uint8_t>> seen(total_ops);
  for (uint64_t i = 0; i < total_ops; ++i) {
    seen[i].store(0, std::memory_order_relaxed);
  }

  std::atomic<uint64_t> produced{0};
  std::atomic<uint64_t> consumed{0};
  std::atomic<int> producers_done{0};
  test_util::DuplicateInfo dup;
  std::atomic<bool> timeout{false};

  auto start_time = std::chrono::steady_clock::now();
  auto deadline = start_time + std::chrono::seconds(cfg.secs + 5);

  std::vector<std::thread> prod_threads;
  prod_threads.reserve(producers);

  for (int p = 0; p < producers; ++p) {
    prod_threads.emplace_back([&, p] {
      std::mt19937 rng(static_cast<uint32_t>(cfg.seed + p));
      std::uniform_int_distribution<int> delay_us(0, 10);

      for (;;) {
        if (timeout.load(std::memory_order_relaxed)) {
          break;
        }
        uint64_t id = produced.fetch_add(1, std::memory_order_relaxed);
        if (id >= total_ops) {
          break;
        }
        while (!q.push(id)) {
          if (std::chrono::steady_clock::now() > deadline) {
            timeout.store(true, std::memory_order_relaxed);
            break;
          }
          std::this_thread::yield();
        }
        if (timeout.load(std::memory_order_relaxed)) {
          break;
        }
        if (delay_us(rng) > 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(delay_us(rng)));
        }
      }
      producers_done.fetch_add(1, std::memory_order_relaxed);
    });
  }

  for (int c = 0; c < consumers; ++c) {
    pool.post([&, c] {
      std::mt19937 rng(static_cast<uint32_t>(cfg.seed + 1000 + c));
      std::uniform_int_distribution<int> delay_us(0, 10);
      uint64_t id = 0;

      for (;;) {
        if (std::chrono::steady_clock::now() > deadline) {
          timeout.store(true, std::memory_order_relaxed);
          break;
        }
        if (q.pop(id)) {
          if (id < total_ops) {
            uint8_t prev = seen[id].fetch_add(1, std::memory_order_relaxed);
            if (prev != 0) {
              test_util::record_duplicate(dup, id);
            }
          }
          consumed.fetch_add(1, std::memory_order_relaxed);
          if (delay_us(rng) > 0) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(delay_us(rng)));
          }
        } else {
          if (producers_done.load(std::memory_order_relaxed) == producers &&
              consumed.load(std::memory_order_relaxed) >= total_ops) {
            break;
          }
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& t : prod_threads) {
    t.join();
  }

  pool.join();

  EXPECT_FALSE(timeout.load(std::memory_order_relaxed))
      << "timeout during stress test";
  EXPECT_EQ(consumed.load(std::memory_order_relaxed), total_ops)
      << "produced=" << produced.load(std::memory_order_relaxed);
  EXPECT_EQ(dup.count.load(std::memory_order_relaxed), 0u)
      << "first duplicate id=" << dup.first_id.load(std::memory_order_relaxed);

  uint64_t missing = std::numeric_limits<uint64_t>::max();
  for (uint64_t i = 0; i < total_ops; ++i) {
    if (seen[i].load(std::memory_order_relaxed) == 0) {
      missing = i;
      break;
    }
  }
  EXPECT_EQ(missing, std::numeric_limits<uint64_t>::max())
      << "missing id=" << missing;
  EXPECT_LE(q.size(), 1u) << "queue not empty, size=" << q.size();
}
