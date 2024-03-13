#pragma once

#include "../utilities/are_reals_equal.h"
#include "../utilities/get_num_blocks.h"
#include "../utilities/get_max_from_array.cuh"
#include "../utilities/get_mean_from_array.cuh"
#include "../utilities/compute_max_error.cuh"
#include "../utilities/compare_array_on_device_vs_host_real.cuh"
#include "../utilities/compare_array_with_file_bool.h"
#include "../utilities/compare_array_with_file_real.h"
#include "../utilities/compare_d_array_with_file_bool.cuh"
#include "../utilities/compare_d_array_with_file_real.cuh"
#include "../utilities/zero_array.cuh"

void unit_test_get_max_from_array();
void unit_test_get_mean_from_array();
void unit_test_compute_max_error();
void unit_test_compare_array_on_device_vs_host_real();
void unit_test_compare_array_with_file_bool();
void unit_test_compare_array_with_file_real();
void unit_test_compare_d_array_with_file_bool_NO_OFFSET();
void unit_test_compare_d_array_with_file_bool_OFFSET();
void unit_test_compare_d_array_with_file_real_NO_OFFSET();
void unit_test_compare_d_array_with_file_real_OFFSET();
void unit_test_zero_array();
void run_unit_tests_utilities();