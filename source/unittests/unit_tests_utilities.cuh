#pragma once

#include "../utilities/are_reals_equal.h"
#include "../utilities/get_mean_from_array.cuh"
#include "../utilities/compute_error.cuh"
#include "../utilities/compare_array_on_device_vs_host_real.cuh"
#include "../utilities/compare_array_with_file_real.h"

void unit_test_get_mean_from_array();

void unit_test_compute_error();

void unit_test_compare_array_on_device_vs_host_real();

void unit_test_compare_array_with_file_real();

void run_unit_tests_utilities();