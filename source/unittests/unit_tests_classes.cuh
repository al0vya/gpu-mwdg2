#pragma once

#include "../classes/ScaleCoefficients.h"
#include "../utilities/compare_array_with_file_real.h"
#include "../utilities/compare_array_on_device_vs_host_real.cuh"

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_HW();
void unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW();
void unit_test_scale_coeffs_CONSTRUCTOR_COPY_HW();
void unit_test_scale_coeffs_WRITE_TO_FILE_HW();
void unit_test_scale_coeffs_VERIFY_HW();

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_MW();

void run_unit_tests_classes();