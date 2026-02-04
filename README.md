# bthpool

A lightweight, modern C++ thread pool focused on safety, simplicity, and performance. This README shows how to build, use, and understand the design of `bthpool`.

## Quick Start

### Build and run examples/tests

```zsh
# From project root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBTHPOOL_BUILD_EXAMPLES=ON
cmake --build build -j

# Run examples
./build/examples/example_basic
./build/examples/example_advanced
./build/examples/example_concurrent
./build/examples/example_error_handling

# Run tests
ctest --test-dir build --output-on-failure
```

### Basic usage

```cpp
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
```

### Fire-and-forget tasks

```cpp
pool.post([] { /* do work without a return value */ });
```

## API Overview

- `BThreadPoolParam`: Configuration (e.g., `core_thread_num`, `max_thread_num`, `fast_queue_capacity`, `thread_clean_interval`).
- `BThreadPool()`, `BThreadPool(BThreadPoolParam)`: Create a pool with default or custom parameters.
- `post(F&&, Args&&...)`: Fire-and-forget submission; return values are discarded. Exceptions are ignored.
- `defer(F&&, Args&&...)`: Enqueue directly to the slow queue.
- `dispatch(F&&, Args&&...)`: Execute inline when called from a worker; otherwise enqueue.
- `futured_post(F&&, Args&&...) -> std::future<R>`: Submit a task and capture result/exception.
- `join()`: Graceful stop; waits for all queued tasks to finish.
- `shutdown()`: Immediate stop; queued tasks may be dropped.
- `restart()`: Transition from STOPPED back to RUNNING.

## Usage Examples

- [examples/example_basic.cpp](examples/example_basic.cpp): Minimal `BThreadPool` usage and a single `futured_post` result.
- [examples/example_advanced.cpp](examples/example_advanced.cpp): Parameter tuning plus mixed `post` and `defer` usage.
- [examples/example_concurrent.cpp](examples/example_concurrent.cpp): Multiple producer threads posting work concurrently.
- [examples/example_error_handling.cpp](examples/example_error_handling.cpp): Exception capture via `futured_post`.
- [examples/example.cpp](examples/example.cpp): Legacy demo retained for reference.

## Core Concepts

- `BThreadPoolParam` controls thread sizing and queue behavior. If `core_thread_num` is `0` (possible when `std::thread::hardware_concurrency()` returns `0`), set it explicitly.
- `post` and `defer` discard return values; use `futured_post` to observe results or exceptions.
- `dispatch` may run inline when called from a pool worker, which can reduce scheduling overhead.

## Design Highlights

## Threading Model

The pool auto-sizes workers with a model similar to Java's `ThreadPoolExecutor`:

- **Core size**: Defaults to `std::thread::hardware_concurrency()` (may be `0` on some platforms).
- **Max size**: Defaults to effectively unbounded (`param.max_thread_num` set to `INT_MAX`).
- **Fast vs slow queues**: Tasks are first pushed to a lock-free fast queue; if saturated, a worker is spawned up to max size, and the task falls back to the slow queue.
- **Grow on demand**: When posting, if `living_thread_num < core_thread_num`, workers are created immediately. If the fast queue is full and all core threads are busy, the pool can create additional threads up to `max_thread_num`.
- **Shrink when idle**: Idle workers opportunistically clean themselves up to return to the core size. Threads above `core_thread_num` decrement `living_thread_num` and schedule a cleaner to stop and join the worker.
- **Blocking strategy**: Workers wait on a condition variable and wake on new tasks or shutdown.
- **Shutdown semantics**: `join()` transitions to STOPPING and waits for workers to finish; `shutdown()` transitions to STOPPED and stops without draining.

In practice:

- Under light load, the pool maintains ~core threads.
- Under pressure, it tries the fast queue first, then grows threads and uses the slow queue to absorb bursty tasks.
- When load subsides, excess workers are cleaned and the pool returns to core size.

Source references: see `include/bthpool/bthpool.hpp` — `BThreadPoolParam` (core/max), `post()` (growth), `ThreadWorkerFunctor::try_cleanup()` (shrink), and the two queues in the private state.

## Tips & FAQs

- Prefer `futured_post` when you need results or exceptions; use `post`/`defer` for side-effect-only work.
- If you need to throttle producers, consider adding a bounded queue or back-pressure.
- Avoid capturing large objects by copy in lambdas; use references or `std::shared_ptr`.
- If `std::thread::hardware_concurrency()` returns `0`, explicitly set `core_thread_num` in `BThreadPoolParam`.

## Build & Run Examples

```zsh
# Configure and build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBTHPOOL_BUILD_EXAMPLES=ON
cmake --build build -j

# Run individual examples
./build/examples/example_basic
./build/examples/example_advanced
./build/examples/example_concurrent
./build/examples/example_error_handling
```

## Project Layout

- `include/bthpool/bthpool.hpp`: Public thread pool API.
- `include/bthpool/internal/safe_queue.hpp`: Thread-safe queue used by the pool.
- `examples/`: Example programs.
- `tests/`: Basic tests.
- `cmake/`: CMake config templates.
