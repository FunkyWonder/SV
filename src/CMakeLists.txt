cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0104 NEW)

project(SV-file VERSION 1.0)

set(CMAKE_BUILD_TYPE Debug)

enable_language(C CUDA) ## Was C CUDA

# Generates a `compile_commands.json` that can be used for autocompletion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)

set(OSBitness 32)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OSBitness 64)
endif()

set (FullOutputDir "${CMAKE_SOURCE_DIR}/../build/${CMAKE_SYSTEM_NAME}${OSBitness}/${CMAKE_BUILD_TYPE}")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${FullOutputDir}/static libs")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${FullOutputDir}")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${FullOutputDir}")

include(CTest)

set(SOURCES main.cu main.h)

add_library(SV-library STATIC
  ${SOURCES}
)

target_compile_features(SV-library PUBLIC cxx_std_14)

set_target_properties( SV-library
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties( SV-library
  PROPERTIES CUDA_ARCHITECTURES "86")

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--diag-suppress=177,550")

add_executable(SV-file ${SOURCES})

target_compile_options(SV-file PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-G> 
  $<$<COMPILE_LANGUAGE:CUDA>:-dc>) 

target_link_libraries(SV-file PRIVATE SV-library)

if(APPLE)
  set_property(TARGET SV-file
    PROPERTY
    BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
