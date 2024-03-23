#include "unit_tests_classes.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params);

	bool passed =
	(
		d_assem_sol.is_copy_cuda == false &&
		d_assem_sol.max_length   == 1 << (2 * solver_params.L) &&
		d_assem_sol.length       == 1 << (2 * solver_params.L) &&
		d_assem_sol.solver_type  == solver_params.solver_type &&
		d_assem_sol.name         == "" &&
		d_assem_sol.h0           != nullptr &&
		d_assem_sol.qx0          != nullptr &&
		d_assem_sol.qy0          != nullptr &&
		d_assem_sol.z0           != nullptr &&
		d_assem_sol.h1x          == nullptr &&
		d_assem_sol.qx1x         == nullptr &&
		d_assem_sol.qy1x         == nullptr &&
		d_assem_sol.z1x          == nullptr &&
		d_assem_sol.h1y          == nullptr &&
		d_assem_sol.qx1y         == nullptr &&
		d_assem_sol.qy1y         == nullptr &&
		d_assem_sol.z1y          == nullptr &&
		d_assem_sol.act_idcs     != nullptr &&
		d_assem_sol.levels       != nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_HW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_HW";

	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params, dirroot, prefix);

	const int array_length = 1 << (2 * solver_params.L);
	real* h_assem_sol_real = new real[array_length];
	int*  h_assem_sol_int  = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	const real error_h0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h0,       array_length);
	const real error_qx0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx0,      array_length);
	const real error_qy0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy0,      array_length);
	const real error_z0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z0,       array_length);
	const int  diffs_levels   = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.levels,   array_length);
	const int  diffs_act_idcs = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.act_idcs, array_length);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,  expected_error) &&
		are_reals_equal(error_qx0, expected_error) &&
		are_reals_equal(error_qy0, expected_error) &&
		are_reals_equal(error_z0,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params);

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];
	
	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i] = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_HW";

	d_assem_sol.write_to_file(dirroot, prefix);
	
	char filename_h0[255]       = {'\0'};
	char filename_qx0[255]      = {'\0'};
	char filename_qy0[255]      = {'\0'};
	char filename_z0[255]       = {'\0'};
	char filename_levels[255]   = {'\0'};
	char filename_act_idcs[255] = {'\0'};

	sprintf(filename_h0,       "%s%s", prefix, "-assem-sol-h0-hw");
	sprintf(filename_qx0,      "%s%s", prefix, "-assem-sol-qx0-hw");
	sprintf(filename_qy0,      "%s%s", prefix, "-assem-sol-qy0-hw");
	sprintf(filename_z0,       "%s%s", prefix, "-assem-sol-z0-hw");
	sprintf(filename_levels,   "%s%s", prefix, "-assem-sol-levels-hw");
	sprintf(filename_act_idcs, "%s%s", prefix, "-assem-sol-act-idcs-hw");

	const real error_h0       = compare_array_with_file_real(dirroot, filename_h0,       h_assem_sol_real, array_length);
	const real error_qx0      = compare_array_with_file_real(dirroot, filename_qx0,      h_assem_sol_real, array_length);
	const real error_qy0      = compare_array_with_file_real(dirroot, filename_qy0,      h_assem_sol_real, array_length);
	const real error_z0       = compare_array_with_file_real(dirroot, filename_z0,       h_assem_sol_real, array_length);
	const int  diffs_levels   = compare_array_with_file_int (dirroot, filename_levels,   h_assem_sol_int,  array_length);
	const int  diffs_act_idcs = compare_array_with_file_int (dirroot, filename_act_idcs, h_assem_sol_int,  array_length);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,  expected_error) &&
		are_reals_equal(error_qx0, expected_error) &&
		are_reals_equal(error_qy0, expected_error) &&
		are_reals_equal(error_z0,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_VERIFY_NO_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params);

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_NO_NAME_HW";

	const int  diffs          = d_assem_sol.verify_int(dirroot, prefix);
	const real actual_error   = d_assem_sol.verify_real(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (are_reals_equal(actual_error, expected_error) && diffs == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	bool passed =
	(
		d_assem_sol.is_copy_cuda == false &&
		d_assem_sol.length       == 1 << (2 * solver_params.L) &&
		d_assem_sol.max_length   == 1 << (2 * solver_params.L) &&
		d_assem_sol.solver_type  == solver_params.solver_type &&
		d_assem_sol.name         == "-dummy" &&
		d_assem_sol.h0           != nullptr &&
		d_assem_sol.qx0          != nullptr &&
		d_assem_sol.qy0          != nullptr &&
		d_assem_sol.z0           != nullptr &&
		d_assem_sol.h1x          == nullptr &&
		d_assem_sol.qx1x         == nullptr &&
		d_assem_sol.qy1x         == nullptr &&
		d_assem_sol.z1x          == nullptr &&
		d_assem_sol.h1y          == nullptr &&
		d_assem_sol.qx1y         == nullptr &&
		d_assem_sol.qy1y         == nullptr &&
		d_assem_sol.z1y          == nullptr &&
		d_assem_sol.act_idcs     != nullptr &&
		d_assem_sol.levels       != nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_HW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_HW";

	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params, "dummy", dirroot, prefix);

	const int array_length = 1 << (2 * solver_params.L);
	real* h_assem_sol_real = new real[array_length];
	int*  h_assem_sol_int  = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	const real error_h0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h0,       array_length);
	const real error_qx0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx0,      array_length);
	const real error_qy0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy0,      array_length);
	const real error_z0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z0,       array_length);
	const int  diffs_levels   = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.levels,   array_length);
	const int  diffs_act_idcs = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.act_idcs, array_length);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,  expected_error) &&
		are_reals_equal(error_qx0, expected_error) &&
		are_reals_equal(error_qy0, expected_error) &&
		are_reals_equal(error_z0,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];
	
	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i] = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_HW";

	d_assem_sol.write_to_file(dirroot, prefix);
	
	char filename_h0[255]       = {'\0'};
	char filename_qx0[255]      = {'\0'};
	char filename_qy0[255]      = {'\0'};
	char filename_z0[255]       = {'\0'};
	char filename_levels[255]   = {'\0'};
	char filename_act_idcs[255] = {'\0'};

	sprintf(filename_h0,       "%s%s", prefix, "-dummy-assem-sol-h0-hw");
	sprintf(filename_qx0,      "%s%s", prefix, "-dummy-assem-sol-qx0-hw");
	sprintf(filename_qy0,      "%s%s", prefix, "-dummy-assem-sol-qy0-hw");
	sprintf(filename_z0,       "%s%s", prefix, "-dummy-assem-sol-z0-hw");
	sprintf(filename_levels,   "%s%s", prefix, "-dummy-assem-sol-levels-hw");
	sprintf(filename_act_idcs, "%s%s", prefix, "-dummy-assem-sol-act-idcs-hw");

	const real error_h0       = compare_array_with_file_real(dirroot, filename_h0,       h_assem_sol_real, array_length);
	const real error_qx0      = compare_array_with_file_real(dirroot, filename_qx0,      h_assem_sol_real, array_length);
	const real error_qy0      = compare_array_with_file_real(dirroot, filename_qy0,      h_assem_sol_real, array_length);
	const real error_z0       = compare_array_with_file_real(dirroot, filename_z0,       h_assem_sol_real, array_length);
	const int  diffs_levels   = compare_array_with_file_int (dirroot, filename_levels,   h_assem_sol_int,  array_length);
	const int  diffs_act_idcs = compare_array_with_file_int (dirroot, filename_act_idcs, h_assem_sol_int,  array_length);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,  expected_error) &&
		are_reals_equal(error_qx0, expected_error) &&
		are_reals_equal(error_qy0, expected_error) &&
		are_reals_equal(error_z0,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_VERIFY_WITH_NAME_HW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = HWFV1;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_WITH_NAME_HW";

	const int  diffs          = d_assem_sol.verify_int(dirroot, prefix);
	const real actual_error   = d_assem_sol.verify_real(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (are_reals_equal(actual_error, expected_error) && diffs == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params);

	bool passed =
	(
		d_assem_sol.is_copy_cuda == false &&
		d_assem_sol.length       == 1 << (2 * solver_params.L) &&
		d_assem_sol.max_length   == 1 << (2 * solver_params.L) &&
		d_assem_sol.solver_type  == solver_params.solver_type &&
		d_assem_sol.name         == "" &&
		d_assem_sol.h0           != nullptr &&
		d_assem_sol.qx0          != nullptr &&
		d_assem_sol.qy0          != nullptr &&
		d_assem_sol.z0           != nullptr &&
		d_assem_sol.h1x          != nullptr &&
		d_assem_sol.qx1x         != nullptr &&
		d_assem_sol.qy1x         != nullptr &&
		d_assem_sol.z1x          != nullptr &&
		d_assem_sol.h1y          != nullptr &&
		d_assem_sol.qx1y         != nullptr &&
		d_assem_sol.qy1y         != nullptr &&
		d_assem_sol.z1y          != nullptr &&
		d_assem_sol.act_idcs     != nullptr &&
		d_assem_sol.levels       != nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_MW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_MW";

	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params, dirroot, prefix);

	const int array_length = 1 << (2 * solver_params.L);
	real* h_assem_sol_real = new real[array_length];
	int*  h_assem_sol_int  = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	const real error_h0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h0,       array_length);
	const real error_qx0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx0,      array_length);
	const real error_qy0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy0,      array_length);
	const real error_z0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z0,       array_length);
	const real error_h1x      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h1x,       array_length);
	const real error_qx1x     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx1x,      array_length);
	const real error_qy1x     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy1x,      array_length);
	const real error_z1x      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z1x,       array_length);
	const real error_h1y      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h1y,       array_length);
	const real error_qx1y     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx1y,      array_length);
	const real error_qy1y     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy1y,      array_length);
	const real error_z1y      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z1y,       array_length);
	const int  diffs_levels   = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.levels,   array_length);
	const int  diffs_act_idcs = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.act_idcs, array_length);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,   expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error) &&
		are_reals_equal(error_h1x,  expected_error) &&
		are_reals_equal(error_qx1x, expected_error) &&
		are_reals_equal(error_qy1x, expected_error) &&
		are_reals_equal(error_z1x,  expected_error) &&
		are_reals_equal(error_h1y,  expected_error) &&
		are_reals_equal(error_qx1y, expected_error) &&
		are_reals_equal(error_qy1y, expected_error) &&
		are_reals_equal(error_z1y,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params);

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];
	
	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i] = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_NO_NAME_MW";

	d_assem_sol.write_to_file(dirroot, prefix);
	
	char filename_h0[255]       = {'\0'};
	char filename_qx0[255]      = {'\0'};
	char filename_qy0[255]      = {'\0'};
	char filename_z0[255]       = {'\0'};
	char filename_h1x[255]      = {'\0'};
	char filename_qx1x[255]     = {'\0'};
	char filename_qy1x[255]     = {'\0'};
	char filename_z1x[255]      = {'\0'};
	char filename_h1y[255]      = {'\0'};
	char filename_qx1y[255]     = {'\0'};
	char filename_qy1y[255]     = {'\0'};
	char filename_z1y[255]      = {'\0'};
	char filename_levels[255]   = {'\0'};
	char filename_act_idcs[255] = {'\0'};

	sprintf(filename_h0,       "%s%s", prefix, "-assem-sol-h0-mw");
	sprintf(filename_qx0,      "%s%s", prefix, "-assem-sol-qx0-mw");
	sprintf(filename_qy0,      "%s%s", prefix, "-assem-sol-qy0-mw");
	sprintf(filename_z0,       "%s%s", prefix, "-assem-sol-z0-mw");
	sprintf(filename_h1x,      "%s%s", prefix, "-assem-sol-h1x-mw");
	sprintf(filename_qx1x,     "%s%s", prefix, "-assem-sol-qx1x-mw");
	sprintf(filename_qy1x,     "%s%s", prefix, "-assem-sol-qy1x-mw");
	sprintf(filename_z1x,      "%s%s", prefix, "-assem-sol-z1x-mw");
	sprintf(filename_h1y,      "%s%s", prefix, "-assem-sol-h1y-mw");
	sprintf(filename_qx1y,     "%s%s", prefix, "-assem-sol-qx1y-mw");
	sprintf(filename_qy1y,     "%s%s", prefix, "-assem-sol-qy1y-mw");
	sprintf(filename_z1y,      "%s%s", prefix, "-assem-sol-z1y-mw");
	sprintf(filename_levels,   "%s%s", prefix, "-assem-sol-levels-mw");
	sprintf(filename_act_idcs, "%s%s", prefix, "-assem-sol-act-idcs-mw");

	const real error_h0       = compare_array_with_file_real(dirroot, filename_h0,       h_assem_sol_real, array_length);
	const real error_qx0      = compare_array_with_file_real(dirroot, filename_qx0,      h_assem_sol_real, array_length);
	const real error_qy0      = compare_array_with_file_real(dirroot, filename_qy0,      h_assem_sol_real, array_length);
	const real error_z0       = compare_array_with_file_real(dirroot, filename_z0,       h_assem_sol_real, array_length);
	const real error_h1x      = compare_array_with_file_real(dirroot, filename_h1x,      h_assem_sol_real, array_length);
	const real error_qx1x     = compare_array_with_file_real(dirroot, filename_qx1x,     h_assem_sol_real, array_length);
	const real error_qy1x     = compare_array_with_file_real(dirroot, filename_qy1x,     h_assem_sol_real, array_length);
	const real error_z1x      = compare_array_with_file_real(dirroot, filename_z1x,      h_assem_sol_real, array_length);
	const real error_h1y      = compare_array_with_file_real(dirroot, filename_h1y,      h_assem_sol_real, array_length);
	const real error_qx1y     = compare_array_with_file_real(dirroot, filename_qx1y,     h_assem_sol_real, array_length);
	const real error_qy1y     = compare_array_with_file_real(dirroot, filename_qy1y,     h_assem_sol_real, array_length);
	const real error_z1y      = compare_array_with_file_real(dirroot, filename_z1y,      h_assem_sol_real, array_length);
	const int  diffs_levels   = compare_array_with_file_int (dirroot, filename_levels,   h_assem_sol_int,  array_length);
	const int  diffs_act_idcs = compare_array_with_file_int (dirroot, filename_act_idcs, h_assem_sol_int,  array_length);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,   expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error) &&
		are_reals_equal(error_h1x,  expected_error) &&
		are_reals_equal(error_qx1x, expected_error) &&
		are_reals_equal(error_qy1x, expected_error) &&
		are_reals_equal(error_z1x,  expected_error) &&
		are_reals_equal(error_h1y,  expected_error) &&
		are_reals_equal(error_qx1y, expected_error) &&
		are_reals_equal(error_qy1y, expected_error) &&
		are_reals_equal(error_z1y,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_VERIFY_NO_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params);

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_NO_NAME_MW";

	const int  diffs          = d_assem_sol.verify_int(dirroot, prefix);
	const real actual_error   = d_assem_sol.verify_real(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (are_reals_equal(actual_error, expected_error) && diffs == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	bool passed =
	(
		d_assem_sol.is_copy_cuda == false &&
		d_assem_sol.length       == 1 << (2 * solver_params.L) &&
		d_assem_sol.max_length   == 1 << (2 * solver_params.L) &&
		d_assem_sol.solver_type  == solver_params.solver_type &&
		d_assem_sol.name         == "-dummy" &&
		d_assem_sol.h0           != nullptr &&
		d_assem_sol.qx0          != nullptr &&
		d_assem_sol.qy0          != nullptr &&
		d_assem_sol.z0           != nullptr &&
		d_assem_sol.h1x          != nullptr &&
		d_assem_sol.qx1x         != nullptr &&
		d_assem_sol.qy1x         != nullptr &&
		d_assem_sol.z1x          != nullptr &&
		d_assem_sol.h1y          != nullptr &&
		d_assem_sol.qx1y         != nullptr &&
		d_assem_sol.qy1y         != nullptr &&
		d_assem_sol.z1y          != nullptr &&
		d_assem_sol.act_idcs     != nullptr &&
		d_assem_sol.levels       != nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_MW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_MW";

	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params, "dummy", dirroot, prefix);

	const int array_length = 1 << (2 * solver_params.L);
	real* h_assem_sol_real = new real[array_length];
	int*  h_assem_sol_int  = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	const real error_h0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h0,       array_length);
	const real error_qx0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx0,      array_length);
	const real error_qy0      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy0,      array_length);
	const real error_z0       = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z0,       array_length);
	const real error_h1x      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h1x,       array_length);
	const real error_qx1x     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx1x,      array_length);
	const real error_qy1x     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy1x,      array_length);
	const real error_z1x      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z1x,       array_length);
	const real error_h1y      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.h1y,       array_length);
	const real error_qx1y     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qx1y,      array_length);
	const real error_qy1y     = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.qy1y,      array_length);
	const real error_z1y      = compare_array_on_device_vs_host_real(h_assem_sol_real, d_assem_sol.z1y,       array_length);
	const int  diffs_levels   = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.levels,   array_length);
	const int  diffs_act_idcs = compare_array_on_device_vs_host_int (h_assem_sol_int,  d_assem_sol.act_idcs, array_length);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,   expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error) &&
		are_reals_equal(error_h1x,  expected_error) &&
		are_reals_equal(error_qx1x, expected_error) &&
		are_reals_equal(error_qy1x, expected_error) &&
		are_reals_equal(error_z1x,  expected_error) &&
		are_reals_equal(error_h1y,  expected_error) &&
		are_reals_equal(error_qx1y, expected_error) &&
		are_reals_equal(error_qy1y, expected_error) &&
		are_reals_equal(error_z1y,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];
	
	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i] = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_WITH_NAME_MW";

	d_assem_sol.write_to_file(dirroot, prefix);
	
	char filename_h0[255]       = {'\0'};
	char filename_qx0[255]      = {'\0'};
	char filename_qy0[255]      = {'\0'};
	char filename_z0[255]       = {'\0'};
	char filename_h1x[255]      = {'\0'};
	char filename_qx1x[255]     = {'\0'};
	char filename_qy1x[255]     = {'\0'};
	char filename_z1x[255]      = {'\0'};
	char filename_h1y[255]      = {'\0'};
	char filename_qx1y[255]     = {'\0'};
	char filename_qy1y[255]     = {'\0'};
	char filename_z1y[255]      = {'\0'};
	char filename_levels[255]   = {'\0'};
	char filename_act_idcs[255] = {'\0'};

	sprintf(filename_h0,       "%s%s", prefix, "-dummy-assem-sol-h0-mw");
	sprintf(filename_qx0,      "%s%s", prefix, "-dummy-assem-sol-qx0-mw");
	sprintf(filename_qy0,      "%s%s", prefix, "-dummy-assem-sol-qy0-mw");
	sprintf(filename_z0,       "%s%s", prefix, "-dummy-assem-sol-z0-mw");
	sprintf(filename_h1x,      "%s%s", prefix, "-dummy-assem-sol-h1x-mw");
	sprintf(filename_qx1x,     "%s%s", prefix, "-dummy-assem-sol-qx1x-mw");
	sprintf(filename_qy1x,     "%s%s", prefix, "-dummy-assem-sol-qy1x-mw");
	sprintf(filename_z1x,      "%s%s", prefix, "-dummy-assem-sol-z1x-mw");
	sprintf(filename_h1y,      "%s%s", prefix, "-dummy-assem-sol-h1y-mw");
	sprintf(filename_qx1y,     "%s%s", prefix, "-dummy-assem-sol-qx1y-mw");
	sprintf(filename_qy1y,     "%s%s", prefix, "-dummy-assem-sol-qy1y-mw");
	sprintf(filename_z1y,      "%s%s", prefix, "-dummy-assem-sol-z1y-mw");
	sprintf(filename_levels,   "%s%s", prefix, "-dummy-assem-sol-levels-mw");
	sprintf(filename_act_idcs, "%s%s", prefix, "-dummy-assem-sol-act-idcs-mw");

	const real error_h0       = compare_array_with_file_real(dirroot, filename_h0,       h_assem_sol_real, array_length);
	const real error_qx0      = compare_array_with_file_real(dirroot, filename_qx0,      h_assem_sol_real, array_length);
	const real error_qy0      = compare_array_with_file_real(dirroot, filename_qy0,      h_assem_sol_real, array_length);
	const real error_z0       = compare_array_with_file_real(dirroot, filename_z0,       h_assem_sol_real, array_length);
	const real error_h1x      = compare_array_with_file_real(dirroot, filename_h1x,      h_assem_sol_real, array_length);
	const real error_qx1x     = compare_array_with_file_real(dirroot, filename_qx1x,     h_assem_sol_real, array_length);
	const real error_qy1x     = compare_array_with_file_real(dirroot, filename_qy1x,     h_assem_sol_real, array_length);
	const real error_z1x      = compare_array_with_file_real(dirroot, filename_z1x,      h_assem_sol_real, array_length);
	const real error_h1y      = compare_array_with_file_real(dirroot, filename_h1y,      h_assem_sol_real, array_length);
	const real error_qx1y     = compare_array_with_file_real(dirroot, filename_qx1y,     h_assem_sol_real, array_length);
	const real error_qy1y     = compare_array_with_file_real(dirroot, filename_qy1y,     h_assem_sol_real, array_length);
	const real error_z1y      = compare_array_with_file_real(dirroot, filename_z1y,      h_assem_sol_real, array_length);
	const int  diffs_levels   = compare_array_with_file_int (dirroot, filename_levels,   h_assem_sol_int,  array_length);
	const int  diffs_act_idcs = compare_array_with_file_int (dirroot, filename_act_idcs, h_assem_sol_int,  array_length);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_h0,   expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error) &&
		are_reals_equal(error_h1x,  expected_error) &&
		are_reals_equal(error_qx1x, expected_error) &&
		are_reals_equal(error_qy1x, expected_error) &&
		are_reals_equal(error_z1x,  expected_error) &&
		are_reals_equal(error_h1y,  expected_error) &&
		are_reals_equal(error_qx1y, expected_error) &&
		are_reals_equal(error_qy1y, expected_error) &&
		are_reals_equal(error_z1y,  expected_error) &&
		diffs_levels   == 0 &&
		diffs_act_idcs == 0
	);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_VERIFY_WITH_NAME_MW()
{
	SolverParams solver_params;

	solver_params.L           = 2;
	solver_params.solver_type = MWDG2;

	AssembledSolution d_assem_sol(solver_params, "dummy");

	const int array_length  = 1 << (2 * solver_params.L);
	const size_t bytes_real = array_length * sizeof(real);
	const size_t bytes_int  = array_length * sizeof(int);
	real* h_assem_sol_real  = new real[array_length];
	int*  h_assem_sol_int   = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_assem_sol_real[i] = i;
		h_assem_sol_int[i]  = i;
	}

	copy_cuda(d_assem_sol.h0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy0,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z0,       h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1x,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1x,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.h1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qx1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.qy1y,     h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.z1y,      h_assem_sol_real, bytes_real);
	copy_cuda(d_assem_sol.levels,   h_assem_sol_int,  bytes_int);
	copy_cuda(d_assem_sol.act_idcs, h_assem_sol_int,  bytes_int);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_assem_sol_VERIFY_WITH_NAME_MW";

	const int  diffs          = d_assem_sol.verify_int(dirroot, prefix);
	const real actual_error   = d_assem_sol.verify_real(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_assem_sol_real;
	delete[] h_assem_sol_int;

	if (are_reals_equal(actual_error, expected_error) && diffs == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_assem_sol_CONSTRUCTOR_COPY()
{
	SolverParams solver_params;

	AssembledSolution d_assem_sol(solver_params);

	AssembledSolution d_assem_sol_copy(d_assem_sol);

	if (d_assem_sol_copy.is_copy_cuda)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_HW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = HWFV1;

	ScaleCoefficients d_scale_coeffs(solver_params);

	bool passed =
	(
		d_scale_coeffs.is_copy_cuda == false &&
		d_scale_coeffs.levels       == solver_params.L &&
		d_scale_coeffs.solver_type  == solver_params.solver_type &&
		d_scale_coeffs.eta0         != nullptr &&
		d_scale_coeffs.qx0          != nullptr &&
		d_scale_coeffs.qy0          != nullptr &&
		d_scale_coeffs.z0           != nullptr &&
		d_scale_coeffs.eta1x        == nullptr &&
		d_scale_coeffs.qx1x         == nullptr &&
		d_scale_coeffs.qy1x         == nullptr &&
		d_scale_coeffs.z1x          == nullptr &&
		d_scale_coeffs.eta1y        == nullptr &&
		d_scale_coeffs.qx1y         == nullptr &&
		d_scale_coeffs.qy1y         == nullptr &&
		d_scale_coeffs.z1y          == nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW";

	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	ScaleCoefficients d_scale_coeffs(solver_params, dirroot, prefix);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	real* h_scale_coeffs = new real[array_length];

	for (int i = 0; i < PADDING_MRA; i++)
	{
		h_scale_coeffs[i] = 0;
	}

	for (int i = 0; i < array_length - PADDING_MRA; i++)
	{
		h_scale_coeffs[i + PADDING_MRA] = i;
	}

	const real error_eta0  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta0, array_length);
	const real error_qx0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx0,  array_length);
	const real error_qy0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy0,  array_length);
	const real error_z0    = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z0,   array_length);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_eta0, expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error)
	);

	delete[] h_scale_coeffs;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_WRITE_TO_FILE_HW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	ScaleCoefficients d_scale_coeffs(solver_params);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	const size_t bytes     = array_length * sizeof(real);
	real* h_scale_coeffs    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = i;
	}

	copy_cuda(d_scale_coeffs.eta0, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z0,   h_scale_coeffs, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_WRITE_TO_FILE_HW";

	d_scale_coeffs.write_to_file(dirroot, prefix);
	
	char filename_eta0[255] = {'\0'};
	char filename_qx0[255]  = {'\0'};
	char filename_qy0[255]  = {'\0'};
	char filename_z0[255]   = {'\0'};

	sprintf(filename_eta0, "%s%s", prefix, "-scale-coeffs-eta0-hw");
	sprintf(filename_qx0,  "%s%s", prefix, "-scale-coeffs-qx0-hw");
	sprintf(filename_qy0,  "%s%s", prefix, "-scale-coeffs-qy0-hw");
	sprintf(filename_z0,   "%s%s", prefix, "-scale-coeffs-z0-hw");

	const real error_eta0 = compare_array_with_file_real(dirroot, filename_eta0, h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qx0  = compare_array_with_file_real(dirroot, filename_qx0,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qy0  = compare_array_with_file_real(dirroot, filename_qy0,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_z0   = compare_array_with_file_real(dirroot, filename_z0,   h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_eta0, expected_error) &&
		are_reals_equal(error_qx0,  expected_error) &&
		are_reals_equal(error_qy0,  expected_error) &&
		are_reals_equal(error_z0,   expected_error)
	);

	delete[] h_scale_coeffs;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_VERIFY_HW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	ScaleCoefficients d_scale_coeffs(solver_params);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	const size_t bytes     = array_length * sizeof(real);
	real* h_scale_coeffs    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	copy_cuda(d_scale_coeffs.eta0, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z0,   h_scale_coeffs, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_VERIFY_HW";

	const real actual_error   = d_scale_coeffs.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_scale_coeffs;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_MW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = MWDG2;

	ScaleCoefficients d_scale_coeffs(solver_params);

	bool passed =
	(
		d_scale_coeffs.is_copy_cuda == false &&
		d_scale_coeffs.levels       == solver_params.L &&
		d_scale_coeffs.solver_type  == solver_params.solver_type &&
		d_scale_coeffs.eta0         != nullptr &&
		d_scale_coeffs.qx0          != nullptr &&
		d_scale_coeffs.qy0          != nullptr &&
		d_scale_coeffs.z0           != nullptr &&
		d_scale_coeffs.eta1x        != nullptr &&
		d_scale_coeffs.qx1x         != nullptr &&
		d_scale_coeffs.qy1x         != nullptr &&
		d_scale_coeffs.z1x          != nullptr &&
		d_scale_coeffs.eta1y        != nullptr &&
		d_scale_coeffs.qx1y         != nullptr &&
		d_scale_coeffs.qy1y         != nullptr &&
		d_scale_coeffs.z1y          != nullptr
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW";

	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	ScaleCoefficients d_scale_coeffs(solver_params, dirroot, prefix);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	real* h_scale_coeffs = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	const real error_eta0  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta0,  array_length);
	const real error_qx0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx0,   array_length);
	const real error_qy0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy0,   array_length);
	const real error_z0    = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z0,    array_length);
	const real error_eta1x = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta1x, array_length);
	const real error_qx1x  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx1x,  array_length);
	const real error_qy1x  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy1x,  array_length);
	const real error_z1x   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z1x,   array_length);
	const real error_eta1y = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta1y, array_length);
	const real error_qx1y  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx1y,  array_length);
	const real error_qy1y  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy1y,  array_length);
	const real error_z1y   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z1y,   array_length);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_eta0,  expected_error) &&
		are_reals_equal(error_qx0,   expected_error) &&
		are_reals_equal(error_qy0,   expected_error) &&
		are_reals_equal(error_z0,    expected_error) &&
		are_reals_equal(error_eta1x, expected_error) &&
		are_reals_equal(error_qx1x,  expected_error) &&
		are_reals_equal(error_qy1x,  expected_error) &&
		are_reals_equal(error_z1x,   expected_error) &&
		are_reals_equal(error_eta1y, expected_error) &&
		are_reals_equal(error_qx1y,  expected_error) &&
		are_reals_equal(error_qy1y,  expected_error) &&
		are_reals_equal(error_z1y,   expected_error) 
	);

	delete[] h_scale_coeffs;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_WRITE_TO_FILE_MW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	ScaleCoefficients d_scale_coeffs(solver_params);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	const size_t bytes     = array_length * sizeof(real);
	real* h_scale_coeffs   = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	copy_cuda(d_scale_coeffs.eta0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx0,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy0,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z0,    h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.eta1x, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx1x,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy1x,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z1x,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.eta1y, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx1y,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy1y,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z1y,   h_scale_coeffs, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_WRITE_TO_FILE_MW";

	d_scale_coeffs.write_to_file(dirroot, prefix);
	
	char filename_eta0[255]  = {'\0'};
	char filename_qx0[255]   = {'\0'};
	char filename_qy0[255]   = {'\0'};
	char filename_z0[255]    = {'\0'};
	char filename_eta1x[255] = {'\0'};
	char filename_qx1x[255]  = {'\0'};
	char filename_qy1x[255]  = {'\0'};
	char filename_z1x[255]   = {'\0'};
	char filename_eta1y[255] = {'\0'};
	char filename_qx1y[255]  = {'\0'};
	char filename_qy1y[255]  = {'\0'};
	char filename_z1y[255]   = {'\0'};

	sprintf(filename_eta0,  "%s%s", prefix, "-scale-coeffs-eta0-mw");
	sprintf(filename_qx0,   "%s%s", prefix, "-scale-coeffs-qx0-mw");
	sprintf(filename_qy0,   "%s%s", prefix, "-scale-coeffs-qy0-mw");
	sprintf(filename_z0,    "%s%s", prefix, "-scale-coeffs-z0-mw");
	sprintf(filename_eta1x, "%s%s", prefix, "-scale-coeffs-eta1x-mw");
	sprintf(filename_qx1x,  "%s%s", prefix, "-scale-coeffs-qx1x-mw");
	sprintf(filename_qy1x,  "%s%s", prefix, "-scale-coeffs-qy1x-mw");
	sprintf(filename_z1x,   "%s%s", prefix, "-scale-coeffs-z1x-mw");
	sprintf(filename_eta1y, "%s%s", prefix, "-scale-coeffs-eta1y-mw");
	sprintf(filename_qx1y,  "%s%s", prefix, "-scale-coeffs-qx1y-mw");
	sprintf(filename_qy1y,  "%s%s", prefix, "-scale-coeffs-qy1y-mw");
	sprintf(filename_z1y,   "%s%s", prefix, "-scale-coeffs-z1y-mw");

	const real error_eta0  = compare_array_with_file_real(dirroot, filename_eta0,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qx0   = compare_array_with_file_real(dirroot, filename_qx0,   h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qy0   = compare_array_with_file_real(dirroot, filename_qy0,   h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_z0    = compare_array_with_file_real(dirroot, filename_z0,    h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_eta1x = compare_array_with_file_real(dirroot, filename_eta1x, h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qx1x  = compare_array_with_file_real(dirroot, filename_qx1x,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qy1x  = compare_array_with_file_real(dirroot, filename_qy1x,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_z1x   = compare_array_with_file_real(dirroot, filename_z1x,   h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_eta1y = compare_array_with_file_real(dirroot, filename_eta1y, h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qx1y  = compare_array_with_file_real(dirroot, filename_qx1y,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_qy1y  = compare_array_with_file_real(dirroot, filename_qy1y,  h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);
	const real error_z1y   = compare_array_with_file_real(dirroot, filename_z1y,   h_scale_coeffs + PADDING_MRA, array_length - PADDING_MRA);

	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_eta0,  expected_error) &&
		are_reals_equal(error_qx0,   expected_error) &&
		are_reals_equal(error_qy0,   expected_error) &&
		are_reals_equal(error_z0,    expected_error) &&
		are_reals_equal(error_eta1x, expected_error) &&
		are_reals_equal(error_qx1x,  expected_error) &&
		are_reals_equal(error_qy1x,  expected_error) &&
		are_reals_equal(error_z1x,   expected_error) &&
		are_reals_equal(error_eta1y, expected_error) &&
		are_reals_equal(error_qx1y,  expected_error) &&
		are_reals_equal(error_qy1y,  expected_error) &&
		are_reals_equal(error_z1y,   expected_error) 
	);

	delete[] h_scale_coeffs;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_VERIFY_MW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	ScaleCoefficients d_scale_coeffs(solver_params);

	const int array_length = get_lvl_idx(solver_params.L + 1);
	const size_t bytes     = array_length * sizeof(real);
	real* h_scale_coeffs    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	copy_cuda(d_scale_coeffs.eta0,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx0,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy0,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z0,    h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.eta1x, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx1x,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy1x,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z1x,   h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.eta1y, h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qx1y,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.qy1y,  h_scale_coeffs, bytes);
	copy_cuda(d_scale_coeffs.z1y,   h_scale_coeffs, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_VERIFY_MW";

	const real actual_error   = d_scale_coeffs.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_scale_coeffs;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_COPY()
{
	SolverParams solver_params;

	ScaleCoefficients d_scale_coeffs(solver_params);

	ScaleCoefficients d_scale_coeffs_copy(d_scale_coeffs);

	if (d_scale_coeffs_copy.is_copy_cuda)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_CONSTRUCTOR_DEFAULT()
{
	SubDetails d_subdetails;

	bool passed =
	(
		d_subdetails.alpha        == nullptr &&
		d_subdetails.beta         == nullptr &&
		d_subdetails.gamma        == nullptr &&
		d_subdetails.levels       == -1 &&
		d_subdetails.is_copy_cuda == false
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_CONSTRUCTOR_LEVELS()
{
	SubDetails d_subdetails(0);

	bool initialised = test_subdetails_CONSTRUCTOR_LEVELS(d_subdetails);

	if (initialised)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_CONSTRUCTOR_FILES()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_subdetails_CONSTRUCTOR_FILES";
	const char* suffix  = "theta";

	const int levels = 3;

	SubDetails d_subdetails(levels, dirroot, prefix, suffix);

	const int num_details = get_lvl_idx(levels + 1);
	real* h_subdetails = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_subdetails[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}
	
	bool passed = test_subdetails_CONSTRUCTOR_FILES
	(
		h_subdetails,
		d_subdetails,
		num_details
	);

	delete[] h_subdetails;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_WRITE_TO_FILE()
{
	const int levels = 3;
	const int num_details = get_lvl_idx(levels + 1);
	const size_t bytes = num_details * sizeof(real);
	real* h_subdetails = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_subdetails[i] = i;
	}

	SubDetails d_subdetails(levels);

	copy_cuda(d_subdetails.alpha, h_subdetails, bytes);
	copy_cuda(d_subdetails.beta , h_subdetails, bytes);
	copy_cuda(d_subdetails.gamma, h_subdetails, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_subdetails_WRITE_TO_FILE";
	const char* suffix  = "theta";

	d_subdetails.write_to_file(dirroot, prefix, suffix);

	bool passed = test_subdetails_WRITE_TO_FILE
	(
		dirroot,
		prefix,
		suffix,
		h_subdetails,
		num_details
	);

	delete[] h_subdetails;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_VERIFY()
{
	const int levels      = 3;
	const int num_details = get_lvl_idx(levels + 1);
	const size_t bytes    = num_details * sizeof(real);
	real* h_subdetails    = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_subdetails[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	SubDetails d_subdetails(levels);

	copy_cuda(d_subdetails.alpha, h_subdetails, bytes);
	copy_cuda(d_subdetails.beta,  h_subdetails, bytes);
	copy_cuda(d_subdetails.gamma, h_subdetails, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_subdetails_VERIFY";
	const char* suffix  = "theta";

	const real actual_error   = d_subdetails.verify(dirroot, prefix, suffix);
	const real expected_error = C(0.0);

	delete[] h_subdetails;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_subdetails_CONSTRUCTOR_COPY()
{
	SubDetails d_subdetails;

	SubDetails d_subdetails_copy(d_subdetails);

	if (d_subdetails_copy.is_copy_cuda)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_CONSTRUCTOR_LEVELS_HW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = HWFV1;

	Details d_details(solver_params);

	bool init_eta0  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta0);
	bool init_qx0   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx0);
	bool init_qy0   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy0);
	bool init_z0    = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z0);
	
	bool init_eta1x = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta1x);
	bool init_qx1x  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx1x);
	bool init_qy1x  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy1x);
	bool init_z1x   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z1x);
	
	bool init_eta1y = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta1y);
	bool init_qx1y  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx1y);
	bool init_qy1y  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy1y);
	bool init_z1y   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z1y);

	bool subdetails_correctly_initialised =
	(
		init_eta0   && init_qx0   && init_qy0   && init_z0 &&
		!init_eta1x && !init_qx1x && !init_qy1x && !init_z1x &&
		!init_eta1y && !init_qx1y && !init_qy1y && !init_z1y
	);

	if (subdetails_correctly_initialised && d_details.solver_type == solver_params.solver_type)
		TEST_MESSAGE_PASSED_ELSE_FAILED

}

void unit_test_details_CONSTRUCTOR_FILES_HW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_CONSTRUCTOR_FILES_HW";
	
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	Details d_details(solver_params, dirroot, prefix);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	bool passed_eta0 = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.eta0, num_details);
	bool passed_qx0  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qx0,  num_details);
	bool passed_qy0  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qy0,  num_details);
	bool passed_z0   = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.z0,   num_details);

	bool passed = passed_eta0 && passed_qx0 && passed_qy0 && passed_z0;

	delete[] h_details;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_WRITE_TO_FILE_HW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	Details d_details(solver_params);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	const size_t bytes = num_details * sizeof(real);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	init_details(h_details, d_details, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_WRITE_TO_FILE_HW";

	d_details.write_to_file(dirroot, prefix);

	bool passed_eta0 = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "eta0-hw", h_details, num_details);
	bool passed_qx0  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qx0-hw",  h_details, num_details);
	bool passed_qy0  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qy0-hw",  h_details, num_details);
	bool passed_z0   = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "z0-hw",   h_details, num_details);

	bool passed = passed_eta0 && passed_qx0 && passed_qy0 && passed_z0;

	delete[] h_details;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_VERIFY_HW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	Details d_details(solver_params);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	const size_t bytes = num_details * sizeof(real);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	init_details(h_details, d_details, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_VERIFY_HW";

	const real actual_error = d_details.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_details;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_CONSTRUCTOR_LEVELS_MW()
{
	SolverParams solver_params;

	solver_params.L           = 1;
	solver_params.solver_type = MWDG2;

	Details d_details(solver_params);

	bool init_eta0  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta0);
	bool init_qx0   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx0);
	bool init_qy0   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy0);
	bool init_z0    = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z0);
	
	bool init_eta1x = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta1x);
	bool init_qx1x  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx1x);
	bool init_qy1x  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy1x);
	bool init_z1x   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z1x);
	
	bool init_eta1y = test_subdetails_CONSTRUCTOR_LEVELS(d_details.eta1y);
	bool init_qx1y  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qx1y);
	bool init_qy1y  = test_subdetails_CONSTRUCTOR_LEVELS(d_details.qy1y);
	bool init_z1y   = test_subdetails_CONSTRUCTOR_LEVELS(d_details.z1y);

	bool subdetails_correctly_initialised =
	(
		init_eta0  && init_qx0  && init_qy0  && init_z0 &&
		init_eta1x && init_qx1x && init_qy1x && init_z1x &&
		init_eta1y && init_qx1y && init_qy1y && init_z1y
	);

	if (subdetails_correctly_initialised && d_details.solver_type == solver_params.solver_type)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_CONSTRUCTOR_FILES_MW()
{
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_CONSTRUCTOR_FILES_MW";
	
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	Details d_details(solver_params, dirroot, prefix);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	bool passed_eta0  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.eta0,  num_details);
	bool passed_qx0   = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qx0,   num_details);
	bool passed_qy0   = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qy0,   num_details);
	bool passed_z0    = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.z0,    num_details);
	bool passed_eta1x = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.eta1x, num_details);
	bool passed_qx1x  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qx1x,  num_details);
	bool passed_qy1x  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qy1x,  num_details);
	bool passed_z1x   = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.z1x,   num_details);
	bool passed_eta1y = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.eta1y, num_details);
	bool passed_qx1y  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qx1y,  num_details);
	bool passed_qy1y  = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.qy1y,  num_details);
	bool passed_z1y   = test_subdetails_CONSTRUCTOR_FILES(h_details, d_details.z1y,   num_details);

	bool passed =
	(
		passed_eta0  && passed_qx0  && passed_qy0  && passed_z0 &&
		passed_eta1x && passed_qx1x && passed_qy1x && passed_z1x &&
		passed_eta1y && passed_qx1y && passed_qy1y && passed_z1y
	);

	delete[] h_details;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_WRITE_TO_FILE_MW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	Details d_details(solver_params);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	const size_t bytes = num_details * sizeof(real);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = i;
	}

	init_details(h_details, d_details, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_WRITE_TO_FILE_MW";

	d_details.write_to_file(dirroot, prefix);

	bool passed_eta0  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "eta0-mw",  h_details, num_details);
	bool passed_qx0   = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qx0-mw",   h_details, num_details);
	bool passed_qy0   = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qy0-mw",   h_details, num_details);
	bool passed_z0    = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "z0-mw",    h_details, num_details);
	bool passed_eta1x = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "eta1x-mw", h_details, num_details);
	bool passed_qx1x  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qx1x-mw",  h_details, num_details);
	bool passed_qy1x  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qy1x-mw",  h_details, num_details);
	bool passed_z1x   = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "z1x-mw",   h_details, num_details);
	bool passed_eta1y = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "eta1y-mw", h_details, num_details);
	bool passed_qx1y  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qx1y-mw",  h_details, num_details);
	bool passed_qy1y  = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "qy1y-mw",  h_details, num_details);
	bool passed_z1y   = test_subdetails_WRITE_TO_FILE(dirroot, prefix, "z1y-mw",   h_details, num_details);

	bool passed =
	(
		passed_eta0  && passed_qx0  && passed_qy0  && passed_z0 &&
		passed_eta1x && passed_qx1x && passed_qy1x && passed_z1x &&
		passed_eta1y && passed_qx1y && passed_qy1y && passed_z1y
	);

	delete[] h_details;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_details_VERIFY_MW()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = MWDG2;

	Details d_details(solver_params);

	const int num_details = get_lvl_idx(d_details.eta0.levels + 1);
	const size_t bytes = num_details * sizeof(real);
	real* h_details = new real[num_details];

	for (int i = 0; i < num_details; i++)
	{
		h_details[i] = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA;
	}

	init_details(h_details, d_details, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_details_VERIFY_MW";

	const real actual_error = d_details.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_details;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_classes()
{
	unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_HW();
	unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_HW();
	unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_HW();
	unit_test_assem_sol_VERIFY_NO_NAME_HW();

	unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_HW();
	unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_HW();
	unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_HW();
	unit_test_assem_sol_VERIFY_WITH_NAME_HW();

	unit_test_assem_sol_CONSTRUCTOR_LEVELS_NO_NAME_MW();
	unit_test_assem_sol_CONSTRUCTOR_FILES_NO_NAME_MW();
	unit_test_assem_sol_WRITE_TO_FILE_NO_NAME_MW();
	unit_test_assem_sol_VERIFY_NO_NAME_MW();

	unit_test_assem_sol_CONSTRUCTOR_LEVELS_WITH_NAME_MW();
	unit_test_assem_sol_CONSTRUCTOR_FILES_WITH_NAME_MW();
	unit_test_assem_sol_WRITE_TO_FILE_WITH_NAME_MW();
	unit_test_assem_sol_VERIFY_WITH_NAME_MW();

	unit_test_assem_sol_CONSTRUCTOR_COPY();

	unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_HW();
	unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW();
	unit_test_scale_coeffs_WRITE_TO_FILE_HW();
	unit_test_scale_coeffs_VERIFY_HW();

	unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_MW();
	unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW();
	unit_test_scale_coeffs_WRITE_TO_FILE_MW();
	unit_test_scale_coeffs_VERIFY_MW();

	unit_test_scale_coeffs_CONSTRUCTOR_COPY();

	unit_test_subdetails_CONSTRUCTOR_DEFAULT();
	unit_test_subdetails_CONSTRUCTOR_LEVELS();
	unit_test_subdetails_CONSTRUCTOR_FILES();
	unit_test_subdetails_WRITE_TO_FILE();
	unit_test_subdetails_VERIFY();
	unit_test_subdetails_CONSTRUCTOR_COPY();

	unit_test_details_CONSTRUCTOR_LEVELS_HW();
	unit_test_details_CONSTRUCTOR_FILES_HW();
	unit_test_details_WRITE_TO_FILE_HW();
	unit_test_details_VERIFY_HW();

	unit_test_details_CONSTRUCTOR_LEVELS_MW();
	unit_test_details_CONSTRUCTOR_FILES_MW();
	unit_test_details_WRITE_TO_FILE_MW();
	unit_test_details_VERIFY_MW();
}

bool test_subdetails_CONSTRUCTOR_LEVELS(SubDetails d_subdetails)
{
	bool initialised =
	(
		d_subdetails.alpha        != nullptr &&
		d_subdetails.beta         != nullptr &&
		d_subdetails.gamma        != nullptr &&
		d_subdetails.levels        > -1
	);

	return initialised;
}

bool test_subdetails_CONSTRUCTOR_FILES
(
	real*      h_subdetails,
	SubDetails d_subdetails,
	const int& num_details
)
{
	const real error_alpha = compare_array_on_device_vs_host_real(h_subdetails, d_subdetails.alpha, num_details);
	const real error_beta  = compare_array_on_device_vs_host_real(h_subdetails, d_subdetails.beta,  num_details);
	const real error_gamma = compare_array_on_device_vs_host_real(h_subdetails, d_subdetails.gamma, num_details);
	
	const real expected_error = C(1e-6);

	bool passed =
	(
		are_reals_equal(error_alpha, expected_error) &&
		are_reals_equal(error_beta,  expected_error) &&
		are_reals_equal(error_gamma, expected_error) 
	);

	return passed;
}

bool test_subdetails_WRITE_TO_FILE
(
	const char* dirroot,
	const char* prefix,
	const char* suffix,
	real*       h_subdetails,
	const int&  num_details
)
{
	char filename_alpha[255] = {'\0'};
	char filename_beta[255]  = {'\0'};
	char filename_gamma[255] = {'\0'};

	sprintf(filename_alpha, "%s%s%s", prefix, "-details-alpha-", suffix);
	sprintf(filename_beta,  "%s%s%s", prefix, "-details-beta-",  suffix);
	sprintf(filename_gamma, "%s%s%s", prefix, "-details-gamma-", suffix);

	const real error_alpha = compare_array_with_file_real(dirroot, filename_alpha, h_subdetails + PADDING_MRA, num_details - PADDING_MRA);
	const real error_beta  = compare_array_with_file_real(dirroot, filename_beta,  h_subdetails + PADDING_MRA, num_details - PADDING_MRA);
	const real error_gamma = compare_array_with_file_real(dirroot, filename_gamma, h_subdetails + PADDING_MRA, num_details - PADDING_MRA);
	
	const real expected_error = C(1e-6);

	bool passed = 
	(
		are_reals_equal(error_alpha,  expected_error) &&
		are_reals_equal(error_beta,   expected_error) &&
		are_reals_equal(error_gamma,  expected_error) 
	);

	return passed;
}

void init_details
(
	real*   h_details,
	Details d_details,
	const size_t& bytes
)
{
	if (d_details.solver_type == HWFV1)
	{
		copy_cuda(d_details.eta0.alpha, h_details, bytes);
		copy_cuda(d_details.eta0.beta,  h_details, bytes);
		copy_cuda(d_details.eta0.gamma, h_details, bytes);

		copy_cuda(d_details.qx0.alpha,  h_details, bytes);
		copy_cuda(d_details.qx0.beta,   h_details, bytes);
		copy_cuda(d_details.qx0.gamma,  h_details, bytes);

		copy_cuda(d_details.qy0.alpha,  h_details, bytes);
		copy_cuda(d_details.qy0.beta,   h_details, bytes);
		copy_cuda(d_details.qy0.gamma,  h_details, bytes);

		copy_cuda(d_details.z0.alpha,   h_details, bytes);
		copy_cuda(d_details.z0.beta,    h_details, bytes);
		copy_cuda(d_details.z0.gamma,   h_details, bytes);
	}
	else if (d_details.solver_type == MWDG2)
	{
		copy_cuda(d_details.eta0.alpha, h_details, bytes);
		copy_cuda(d_details.eta0.beta,  h_details, bytes);
		copy_cuda(d_details.eta0.gamma, h_details, bytes);

		copy_cuda(d_details.qx0.alpha,  h_details, bytes);
		copy_cuda(d_details.qx0.beta,   h_details, bytes);
		copy_cuda(d_details.qx0.gamma,  h_details, bytes);

		copy_cuda(d_details.qy0.alpha,  h_details, bytes);
		copy_cuda(d_details.qy0.beta,   h_details, bytes);
		copy_cuda(d_details.qy0.gamma,  h_details, bytes);

		copy_cuda(d_details.z0.alpha,   h_details, bytes);
		copy_cuda(d_details.z0.beta,    h_details, bytes);
		copy_cuda(d_details.z0.gamma,   h_details, bytes);

		copy_cuda(d_details.eta1x.alpha, h_details, bytes);
		copy_cuda(d_details.eta1x.beta,  h_details, bytes);
		copy_cuda(d_details.eta1x.gamma, h_details, bytes);

		copy_cuda(d_details.qx1x.alpha,  h_details, bytes);
		copy_cuda(d_details.qx1x.beta,   h_details, bytes);
		copy_cuda(d_details.qx1x.gamma,  h_details, bytes);

		copy_cuda(d_details.qy1x.alpha,  h_details, bytes);
		copy_cuda(d_details.qy1x.beta,   h_details, bytes);
		copy_cuda(d_details.qy1x.gamma,  h_details, bytes);

		copy_cuda(d_details.z1x.alpha,   h_details, bytes);
		copy_cuda(d_details.z1x.beta,    h_details, bytes);
		copy_cuda(d_details.z1x.gamma,   h_details, bytes);

		copy_cuda(d_details.eta1y.alpha, h_details, bytes);
		copy_cuda(d_details.eta1y.beta,  h_details, bytes);
		copy_cuda(d_details.eta1y.gamma, h_details, bytes);

		copy_cuda(d_details.qx1y.alpha,  h_details, bytes);
		copy_cuda(d_details.qx1y.beta,   h_details, bytes);
		copy_cuda(d_details.qx1y.gamma,  h_details, bytes);

		copy_cuda(d_details.qy1y.alpha,  h_details, bytes);
		copy_cuda(d_details.qy1y.beta,   h_details, bytes);
		copy_cuda(d_details.qy1y.gamma,  h_details, bytes);

		copy_cuda(d_details.z1y.alpha,   h_details, bytes);
		copy_cuda(d_details.z1y.beta,    h_details, bytes);
		copy_cuda(d_details.z1y.gamma,   h_details, bytes);
	}
	else
	{
		fprintf(stderr, "Solver type other than HWFV1 and MWDG2 for Details in %s.\n", __FILE__);
		return;
	}
}

#endif