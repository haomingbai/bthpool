find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )

  # Prefer a system GTest when CI already provides it. Falling back to
  # FetchContent keeps the project self-contained for other environments.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest)
endif()

find_package(Threads REQUIRED)

add_executable(test_bthpool test_basic.cpp)
target_link_libraries(test_bthpool PRIVATE bthpool Threads::Threads)

add_test(NAME basic_run COMMAND test_bthpool)

add_executable(test_threadpool_basic test_threadpool_basic.cpp)
target_link_libraries(test_threadpool_basic PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_threadpool_basic PRIVATE cxx_std_20)

add_executable(test_queue_basic test_queue_basic.cpp)
target_link_libraries(test_queue_basic PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_queue_basic PRIVATE cxx_std_20)

add_executable(test_executor_compat test_executor_compat.cpp)
target_link_libraries(test_executor_compat PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_executor_compat PRIVATE cxx_std_20)

add_executable(test_pmr_allocator test_pmr_allocator.cpp)
target_link_libraries(test_pmr_allocator PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_pmr_allocator PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(test_threadpool_basic)
gtest_discover_tests(test_queue_basic)
gtest_discover_tests(test_executor_compat)
gtest_discover_tests(test_pmr_allocator)
