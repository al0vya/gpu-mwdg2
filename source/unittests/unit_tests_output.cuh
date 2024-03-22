#pragma once

#include "../output/write_d_array_int.cuh"
#include "../output/write_d_array_real.cuh"
#include "../output/write_hierarchy_array_real.cuh"
#include "../output/write_hierarchy_array_bool.cuh"
#include "../utilities/compare_array_with_file_int.h"
#include "../utilities/compare_array_with_file_real.h"

void unit_test_write_hierarchy_array_real();

void unit_test_write_hierarchy_array_bool();

void run_unit_tests_output();