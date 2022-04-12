cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)
enable_language(CUDA)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(proj-gpu-mwdg2)

add_executable(gpu-mwdg2)

include(options.cmake)
include(include.cmake)

add_subdirectory(source)
