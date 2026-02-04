include(FetchContent)

FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

find_package(Threads REQUIRED)

add_executable(test_bthpool test_basic.cpp)
target_link_libraries(test_bthpool PRIVATE bthpool Threads::Threads)

add_test(NAME basic_run COMMAND test_bthpool)

add_executable(test_threadpool_basic test_threadpool_basic.cpp)
target_link_libraries(test_threadpool_basic PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_threadpool_basic PRIVATE cxx_std_23)

add_executable(test_queue_basic test_queue_basic.cpp)
target_link_libraries(test_queue_basic PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_queue_basic PRIVATE cxx_std_23)

add_executable(test_stress test_stress.cpp)
target_link_libraries(test_stress PRIVATE bthpool GTest::gtest_main Threads::Threads)
target_compile_features(test_stress PRIVATE cxx_std_23)

include(GoogleTest)
gtest_discover_tests(test_threadpool_basic)
gtest_discover_tests(test_queue_basic)
gtest_discover_tests(test_stress)
