cmake_minimum_required(VERSION 3.16)

project(bthpool
  VERSION 0.2.4
  DESCRIPTION "A lightweight, modern C++ thread pool focused on safety, simplicity, and performance."
  LANGUAGES CXX)

# Options
option(BTHPOOL_BUILD_EXAMPLES "Build examples" ON)
option(BTHPOOL_BUILD_TESTS "Build tests" ON)
option(BTHPOOL_WITH_BOOST "Enable Boost.Asio integration" ON)

# Library target (header-only)
add_library(bthpool INTERFACE)
add_library(bthpool::bthpool ALIAS bthpool)

target_include_directories(bthpool
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(bthpool INTERFACE cxx_std_23)

target_compile_definitions(bthpool INTERFACE
  $<$<BOOL:${BTHPOOL_WITH_BOOST}>:BTHPOOL_HAS_BOOST_ASIO=1;USE_BOOST_ASIO_EXECUTOR>
  $<$<NOT:$<BOOL:${BTHPOOL_WITH_BOOST}>>:BTHPOOL_HAS_BOOST_ASIO=0>)

if(BTHPOOL_WITH_BOOST)
  find_package(Boost REQUIRED)

  set(_bthpool_boost_targets "")
  if(TARGET Boost::asio)
    list(APPEND _bthpool_boost_targets Boost::asio)
  elseif(TARGET Boost::headers)
    list(APPEND _bthpool_boost_targets Boost::headers)
  elseif(TARGET Boost::boost)
    list(APPEND _bthpool_boost_targets Boost::boost)
  elseif(Boost_INCLUDE_DIRS)
    add_library(Boost::headers INTERFACE IMPORTED)
    set_target_properties(Boost::headers PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIRS}")
    list(APPEND _bthpool_boost_targets Boost::headers)
  endif()

  if(TARGET Boost::system)
    list(APPEND _bthpool_boost_targets Boost::system)
  elseif(Boost_LIBRARIES)
    add_library(Boost::system INTERFACE IMPORTED)
    set_target_properties(Boost::system PROPERTIES
      INTERFACE_LINK_LIBRARIES "${Boost_LIBRARIES}")
    list(APPEND _bthpool_boost_targets Boost::system)
  endif()

  if(_bthpool_boost_targets)
    target_link_libraries(bthpool INTERFACE ${_bthpool_boost_targets})
  else()
    message(FATAL_ERROR "Boost was found but no usable targets or include dirs were detected.")
  endif()
endif()

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
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS bthpool EXPORT bthpoolTargets)

set(BTHPOOL_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/bthpool")

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bthpoolConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfig.cmake
  INSTALL_DESTINATION ${BTHPOOL_CMAKE_INSTALL_DIR}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(EXPORT bthpoolTargets
        NAMESPACE bthpool::
        DESTINATION ${BTHPOOL_CMAKE_INSTALL_DIR})

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/bthpoolConfigVersion.cmake
  DESTINATION ${BTHPOOL_CMAKE_INSTALL_DIR})

# CPack packaging
set(CPACK_PACKAGE_NAME "bthpool")
set(CPACK_PACKAGE_VENDOR "Haoming Bai")
set(CPACK_PACKAGE_CONTACT "Haoming Bai <haomingbai@hotmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "A lightweight, modern C++ thread pool")
set(CPACK_PACKAGE_DESCRIPTION "A lightweight, modern C++ thread pool focused on safety, simplicity, and performance.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/haomingbai/bthpool")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_GENERATOR "TGZ;DEB;RPM")

# Header-only: ship a single development package
set(CPACK_DEBIAN_PACKAGE_NAME "bthpool-dev")
set(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "all")

set(CPACK_RPM_PACKAGE_NAME "bthpool-devel")
set(CPACK_RPM_PACKAGE_ARCHITECTURE "noarch")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")

if(BTHPOOL_WITH_BOOST)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-dev")
  set(CPACK_RPM_PACKAGE_REQUIRES "boost-devel")
endif()

include(CPack)
