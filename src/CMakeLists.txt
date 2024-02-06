cmake_minimum_required(VERSION 3.7)
project(SV-file VERSION 1.0 CXX)

enable_language(CXX)

# Generates a `compile_commands.json` that can be used for autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OSBitness 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSBitness 64)
endif()

set (FullOutputDir "${CMAKE_SOURCE_DIR}/../build/${CMAKE_SYSTEM_NAME}${OSBitness}/${CMAKE_BUILD_TYPE}")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static libs")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

add_executable(SV-file main.c)
target_link_libraries(SV-file m)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit)
  target_link_libraries(SV-file CUDA::cudart)
  target_link_libraries(SV-file CUDA::cuda_driver)

  target_include_directories(SV-file PRIVATE "../include/")
  set(SOURCES "../include/helper_functions.h" "../include/helper_cuda.h")
  target_link_libraries(SV-file ${SOURCES})
else()
  message(STATUS "No CUDA compiler found")
endif()
