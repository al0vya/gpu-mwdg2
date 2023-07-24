cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)
enable_language(CUDA)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# to be able to compile cross platform
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project(proj-gpu-mwdg2)

add_executable(gpu-mwdg2)

include(options.cmake)

add_subdirectory(source)

target_include_directories(gpu-mwdg2 PRIVATE cub)