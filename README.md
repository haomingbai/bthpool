# bthpool

A lightweight, modern C++ thread pool focused on safety, simplicity, and performance. This README shows how to build, use, and understand the design of `bthpool`.

## Quick Start

### Build and run examples/tests

```zsh
# From project root
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run example
./build/examples/example_bthpool

# Run tests
ctest --test-dir build --output-on-failure
```

### Basic usage

```cpp
#include <bthpool/bthpool.hpp>
#include <iostream>

int main() {
    // Create a pool with N threads (defaults to hardware concurrency if 0)
    bthpool::ThreadPool pool(4);

    // Submit tasks; returns std::future<R>
    pool.post([] { return 42; });
    pool.post([](int a, int b) { return a + b; }, 3, 5);

    // Wait for all tasks to complete
    pool.join();

    // Pool shuts down automatically on destruction (graceful join)
    return 0;
}
```

### Fire-and-forget tasks

```cpp
pool.post([] { /* do work without a return value */ });
```

## API Overview

- `ThreadPool(size_t threadCount = 0)`: Creates a pool. `0` chooses `std::thread::hardware_concurrency()`.
- `post(F&& func, Args&&... args) -> std::future<R>`: Enqueue a callable without tracking a result.
- `join()`: Request graceful stop; waits for all workers to finish queued tasks.
- `shutdown()`: Request stop without all workers finishing queued tasks.

Note: Exact names may vary slightly with the header implementation. See `include/bthpool/bthpool.hpp` for authoritative declarations.

## Example

The repo includes an example at `examples/example.cpp`. After building, run `./build/examples/example_bthpool`.

```cpp
#include <bthpool/bthpool.hpp>
#include <chrono>
#include <iostream>

int main() {
    bthpool::ThreadPool pool(4);
    auto slow = [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return 1;
    };

    pool.post(slow);
    pool.post(slow);
    std::cout << "Tasks submitted." << std::endl;
    pool.join();
    std::cout << "All tasks completed." << std::endl;
    return 0;
}
```

## Design Highlights

## Threading Model

The pool auto-sizes workers with a model similar to Java's `ThreadPoolExecutor`:

- **Core size**: Defaults to the number of online CPUs (`get_nproc()`), i.e., `param.core_thread_num`.
- **Max size**: Defaults to effectively unbounded (`param.max_thread_num` set to `INT_MAX`).
- **Fast vs slow queues**: Tasks are first pushed to a lock-free fast queue; if saturated, a worker is spawned up to max size, and the task falls back to the slow queue.
- **Grow on demand**: When posting, if `living_thread_num < core_thread_num`, workers are created immediately. If the fast queue is full and all core threads are busy, the pool can create additional threads up to `max_thread_num`.
- **Shrink when idle**: Idle workers opportunistically clean themselves up to return to the core size. Threads above `core_thread_num` decrement `living_thread_num` and schedule a cleaner to stop and join the worker.
- **Blocking strategy**: Workers wait on a condition variable with a configurable suspend time (`param.suspend_time`) and wake on new tasks or shutdown.
- **Shutdown semantics**: `join()` transitions to STOPPING and waits for workers to finish; `shutdown()` transitions to STOPPED and stops without draining.

In practice:

- Under light load, the pool maintains ~core threads.
- Under pressure, it tries the fast queue first, then grows threads and uses the slow queue to absorb bursty tasks.
- When load subsides, excess workers are cleaned and the pool returns to core size.

Source references: see `include/bthpool/bthpool.hpp` — `BThreadPoolParam` (core/max), `post()` (growth), `ThreadWorkerFunctor::try_cleanup()` (shrink), and the two queues in the private state.

## Tips & FAQs

- Prefer `post` for work that returns values; use `post` for side-effect-only work.
- If you need to throttle producers, consider adding a bounded queue or back-pressure.
- Avoid capturing large objects by copy in lambdas; use references or `std::shared_ptr`.
- If `threadCount == 0`, the pool selects `std::thread::hardware_concurrency()`; this can be `0` on some platforms, in which case a fallback of `1` is used.

## Project Layout

- `include/bthpool/bthpool.hpp`: Public thread pool API.
- `include/bthpool/internal/safe_queue.hpp`: Thread-safe queue used by the pool.
- `examples/`: Example program.
- `tests/`: Basic tests.
- `cmake/`: CMake config templates.
