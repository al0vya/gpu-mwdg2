#pragma once

#include "../classes/ScaleCoefficients.h"
#include "../classes/SubDetails.h"
#include "../utilities/compare_array_with_file_real.h"
#include "../utilities/compare_array_on_device_vs_host_real.cuh"

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_HW();
void unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW();
void unit_test_scale_coeffs_WRITE_TO_FILE_HW();
void unit_test_scale_coeffs_VERIFY_HW();

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_MW();
void unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW();
void unit_test_scale_coeffs_WRITE_TO_FILE_MW();
void unit_test_scale_coeffs_VERIFY_MW();

void unit_test_scale_coeffs_CONSTRUCTOR_COPY();

void unit_test_subdetails_CONSTRUCTOR_DEFAULT();
void unit_test_subdetails_CONSTRUCTOR_LEVELS();
void unit_test_subdetails_CONSTRUCTOR_FILES();
void unit_test_subdetails_WRITE_TO_FILE();
void unit_test_subdetails_VERIFY();
void unit_test_subdetails_CONSTRUCTOR_COPY();

void unit_test_details_CONSTRUCTOR_LEVELS_HW();
void unit_test_details_CONSTRUCTOR_FILES_HW();
void unit_test_details_WRITE_TO_FILE_HW();
void unit_test_details_VERIFY_HW();

void unit_test_details_CONSTRUCTOR_LEVELS_MW();
void unit_test_details_CONSTRUCTOR_FILES_MW();
void unit_test_details_WRITE_TO_FILE_MW();
void unit_test_details_VERIFY_MW();

void run_unit_tests_classes();