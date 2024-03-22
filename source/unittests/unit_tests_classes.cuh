#pragma once

#include "../classes/AssembledSolution.h"
#include "../classes/ScaleCoefficients.h"
#include "../classes/Details.h"
#include "../utilities/compare_array_with_file_int.h"
#include "../utilities/compare_array_with_file_real.h"
#include "../utilities/compare_array_on_device_vs_host_int.cuh"
#include "../utilities/compare_array_on_device_vs_host_real.cuh"

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_HW();
void unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_HW();
void unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_HW();
void unit_test_assem_sol_VERIFY_NO_NAME_HW();

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_HW();
void unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_HW();
void unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_HW();
void unit_test_assem_sol_VERIFY_WITH_NAME_HW();

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_MW();
void unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_MW();
void unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_MW();
void unit_test_assem_sol_VERIFY_NO_NAME_MW();

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_MW();
void unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_MW();
void unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_MW();
void unit_test_assem_sol_VERIFY_WITH_NAME_MW();

void unit_test_assem_sol_CONSTRUCTOR_COPY();

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

bool test_subdetails_CONSTRUCTOR_LEVELS(SubDetails d_subdetails);

bool test_subdetails_CONSTRUCTOR_FILES
(
	real*      h_subdetails,
	SubDetails d_subdetails,
	const int& num_details
);

bool test_subdetails_WRITE_TO_FILE
(
	const char* dirroot,
	const char* prefix,
	const char* suffix,
	real*       h_subdetails,
	const int&  num_details
);

void init_details
(
	real*   h_details,
	Details d_details,
	const   size_t& bytes
);
