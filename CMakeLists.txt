cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)
enable_language(CUDA)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(proj-gpu-mwdg2)

add_executable(gpu-mwdg2)

add_compile_definitions(_USE_TRACER=1)
add_compile_definitions(_USE_DOUBLES=1)

if (CMAKE_CUDA_COMPILER)
	add_subdirectory(source)
	add_subdirectory(source/input)
	add_subdirectory(source/mra)
	add_subdirectory(source/neighbours)
	add_subdirectory(source/operators)
	add_subdirectory(source/output)
	add_subdirectory(source/utilities)
	add_subdirectory(source/zorder)	
endif()

target_include_directories(
        gpu-mwdg2
        PRIVATE
        source/classes
        source/input
        source/mra
        source/neighbours
        source/operators
        source/output
        source/tests
        source/types
        source/utilities
        source/zorder
    )