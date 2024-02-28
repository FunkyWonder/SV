cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0104 NEW)
project(SV-file VERSION 1.0)

set(CMAKE_BUILD_TYPE Debug)

enable_language(C CUDA) ## Was C CUDA
 
# Generates a `compile_commands.json` that can be used for autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 11)

set(OSBitness 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSBitness 64)
endif()

set (FullOutputDir "${CMAKE_SOURCE_DIR}/../build/${CMAKE_SYSTEM_NAME}${OSBitness}/${CMAKE_BUILD_TYPE}")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static libs")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# add dependencies here
CPMAddPackage(
  NAME Thrust
  URL https://github.com/NVIDIA/thrust/archive/refs/tags/2.1.0.zip
)

include(CTest)

set(SOURCES main.cu ../include/helper_cuda.h ../include/helper_string.h)

add_library(SV-library STATIC
  ${SOURCES}
)

target_compile_features(SV-library PUBLIC cxx_std_11)

set_target_properties( SV-library
  PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

set_target_properties( SV-library
  PROPERTIES CUDA_ARCHITECTURES "86")

add_executable(SV-file ${SOURCES})
set_property(TARGET SV-file
  PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

set_property(TARGET SV-file
  PROPERTY CUDA_ARCHITECTURES "86")

target_link_libraries(SV-file PRIVATE SV-library)

if(APPLE)
  set_property(TARGET SV-file
    PROPERTY
    BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
