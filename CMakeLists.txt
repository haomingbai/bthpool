cmake_minimum_required(VERSION 3.16)

project(bthpool
  VERSION 0.1.0
  DESCRIPTION "A lightweight C++ thread pool"
  LANGUAGES CXX)

# Options
option(BTHPOOL_BUILD_EXAMPLES "Build examples" ON)
option(BTHPOOL_BUILD_TESTS "Build tests" ON)

# Library target (header-only)
add_library(bthpool INTERFACE)
target_include_directories(bthpool
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(bthpool INTERFACE cxx_std_17)

# Examples
if(BTHPOOL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Tests
if(BTHPOOL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Installation (optional basic install rules)
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bthpool EXPORT bthpoolTargets)
install(EXPORT bthpoolTargets
        NAMESPACE bthpool::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bthpool)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bthpoolConfig.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfig.cmake @ONLY)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bthpool)
