get_filename_component(BTHPOOL_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(BTHPOOL_CMAKELISTS "${BTHPOOL_ROOT_DIR}/CMakeLists.txt")

if(NOT EXISTS "${BTHPOOL_CMAKELISTS}")
  message(FATAL_ERROR "CMakeLists.txt not found at ${BTHPOOL_CMAKELISTS}")
endif()

file(READ "${BTHPOOL_CMAKELISTS}" _bthpool_cmakelists_content)

string(REGEX MATCH "project\\([^\\)]*VERSION[ \t]+([0-9]+\\.[0-9]+(\\.[0-9]+)?)" _bthpool_version_match "${_bthpool_cmakelists_content}")

if(NOT _bthpool_version_match)
  message(FATAL_ERROR "PROJECT_VERSION not found in ${BTHPOOL_CMAKELISTS}. Ensure project(... VERSION x.y[.z]) is set.")
endif()

set(BTHPOOL_PROJECT_VERSION "${CMAKE_MATCH_1}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E echo "${BTHPOOL_PROJECT_VERSION}"
)