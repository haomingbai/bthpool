// Shared test adapters and utilities.
// Assumptions/Adapter Notes:
// 1) ThreadPool type is bthpool::BThreadPool<>, configurable via bthpool::BThreadPoolParam.
//    If your project exposes a different name or constructor signature, adapt
//    `make_pool()` and the aliases below.
// 2) ThreadPool supports: post/defer/dispatch, futured_post, join, shutdown.
//    If any API differs, adjust the small adapter functions in this file.
// 3) LockfreeFixedQueue and SafeQueue are available in
//    bthpool::internal (inline namespace) and provide push/pop/emplace/size.
//    If your queue lives elsewhere, update the aliases below.
// 4) LockfreeFixedQueue<T> requires T to be trivially destructible and
//    nothrow-constructible; tests use uint64_t.
// 5) Concurrency stress level can be configured via environment variables:
//    - STRESS_SECS (default 3)
//    - STRESS_OPS  (default 500000)
//    - STRESS_THREADS (default min(hardware_concurrency, 8))
//    Random seed is fixed for reproducibility.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <utility>

#include "bthpool/bthpool.hpp"
#include "bthpool/internal/safe_queue.hpp"

namespace test_adapter {
using Pool = bthpool::BThreadPool<>;
using PoolParam = bthpool::BThreadPoolParam;
using bthpool::internal::LockfreeFixedQueue;
using bthpool::internal::SafeQueue;

inline Pool make_pool(std::size_t threads) {
  PoolParam param;
  param.core_thread_num = threads;
  param.max_thread_num = threads;
  // Keep fast queue small to exercise slow-queue fallback by default.
  param.fast_queue_capacity = 32;
  return Pool(param);
}

template <typename P, typename F, typename... Args>
constexpr bool has_futured_post() {
  return requires(P& p, F&& f, Args&&... args) { p.futured_post(std::forward<F>(f), std::forward<Args>(args)...); };
}
}  // namespace test_adapter

namespace test_util {
struct StressConfig {
  int secs;
  int ops;
  int threads;
  uint64_t seed;
};

inline int getenv_int(const char* name, int def_val) {
  if (const char* val = std::getenv(name); val && *val) {
    char* end = nullptr;
    long v = std::strtol(val, &end, 10);
    if (end != val && v > 0 && v < std::numeric_limits<int>::max()) {
      return static_cast<int>(v);
    }
  }
  return def_val;
}

inline StressConfig get_stress_config() {
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  if (hw <= 0) {
    hw = 4;
  }
  int def_threads = (hw < 8) ? hw : 8;
  StressConfig cfg;
  cfg.secs = getenv_int("STRESS_SECS", 3);
  cfg.ops = getenv_int("STRESS_OPS", 500000);
  cfg.threads = getenv_int("STRESS_THREADS", def_threads);
  cfg.seed = 0xC0FFEEu;
  return cfg;
}

struct DuplicateInfo {
  std::atomic<uint64_t> count{0};
  std::atomic<uint64_t> first_id{std::numeric_limits<uint64_t>::max()};
};

inline void record_duplicate(DuplicateInfo& dup, uint64_t id) {
  dup.count.fetch_add(1, std::memory_order_relaxed);
  uint64_t expected = std::numeric_limits<uint64_t>::max();
  dup.first_id.compare_exchange_strong(expected, id, std::memory_order_relaxed);
}
}  // namespace test_util
