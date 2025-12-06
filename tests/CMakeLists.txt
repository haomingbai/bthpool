add_executable(test_bthpool test_basic.cpp)
target_link_libraries(test_bthpool PRIVATE bthpool)

add_test(NAME basic_run COMMAND test_bthpool)
