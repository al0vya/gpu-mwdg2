#include "unit_tests_classes.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

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

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeffs[i] = i;
	}

	bool passed_eta0 = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta0, array_length);
	bool passed_qx0  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx0,  array_length);
	bool passed_qy0  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy0,  array_length);
	bool passed_z0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z0,   array_length);

	bool passed = (passed_eta0 && passed_qx0 && passed_qy0 && passed_z0);

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
	real* h_scale_coeff    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeff[i] = i;
	}

	copy_cuda(d_scale_coeffs.eta0, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z0,   h_scale_coeff, bytes);

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

	bool passed_eta0 = compare_array_with_file_real(dirroot, filename_eta0, h_scale_coeff, array_length);
	bool passed_qx0  = compare_array_with_file_real(dirroot, filename_qx0,  h_scale_coeff, array_length);
	bool passed_qy0  = compare_array_with_file_real(dirroot, filename_qy0,  h_scale_coeff, array_length);
	bool passed_z0   = compare_array_with_file_real(dirroot, filename_z0,   h_scale_coeff, array_length);

	bool passed = (passed_eta0 && passed_qx0 && passed_qy0 && passed_z0);

	delete[] h_scale_coeff;

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
	real* h_scale_coeff    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeff[i] = i;
	}

	copy_cuda(d_scale_coeffs.eta0, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z0,   h_scale_coeff, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_VERIFY_HW";

	const real actual_error   = d_scale_coeffs.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_scale_coeff;

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
		h_scale_coeffs[i] = i;
	}

	bool passed_eta0  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta0,  array_length);
	bool passed_qx0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx0,   array_length);
	bool passed_qy0   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy0,   array_length);
	bool passed_z0    = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z0,    array_length);
	bool passed_eta1x = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta1x, array_length);
	bool passed_qx1x  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx1x,  array_length);
	bool passed_qy1x  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy1x,  array_length);
	bool passed_z1x   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z1x,   array_length);
	bool passed_eta1y = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.eta1y, array_length);
	bool passed_qx1y  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qx1y,  array_length);
	bool passed_qy1y  = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.qy1y,  array_length);
	bool passed_z1y   = compare_array_on_device_vs_host_real(h_scale_coeffs, d_scale_coeffs.z1y,   array_length);

	bool passed =
	(
		passed_eta0  && passed_qx0  && passed_qy0  && passed_z0 &&
		passed_eta1x && passed_qx1x && passed_qy1x && passed_z1x &&
		passed_eta1y && passed_qx1y && passed_qy1y && passed_z1y
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
	real* h_scale_coeff    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeff[i] = i;
	}

	copy_cuda(d_scale_coeffs.eta0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx0,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy0,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z0,    h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.eta1x, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx1x,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy1x,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z1x,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.eta1y, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx1y,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy1y,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z1y,   h_scale_coeff, bytes);

	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW";

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

	bool passed_eta0  = compare_array_with_file_real(dirroot, filename_eta0,  h_scale_coeff, array_length);
	bool passed_qx0   = compare_array_with_file_real(dirroot, filename_qx0,   h_scale_coeff, array_length);
	bool passed_qy0   = compare_array_with_file_real(dirroot, filename_qy0,   h_scale_coeff, array_length);
	bool passed_z0    = compare_array_with_file_real(dirroot, filename_z0,    h_scale_coeff, array_length);
	bool passed_eta1x = compare_array_with_file_real(dirroot, filename_eta1x, h_scale_coeff, array_length);
	bool passed_qx1x  = compare_array_with_file_real(dirroot, filename_qx1x,  h_scale_coeff, array_length);
	bool passed_qy1x  = compare_array_with_file_real(dirroot, filename_qy1x,  h_scale_coeff, array_length);
	bool passed_z1x   = compare_array_with_file_real(dirroot, filename_z1x,   h_scale_coeff, array_length);
	bool passed_eta1y = compare_array_with_file_real(dirroot, filename_eta1y, h_scale_coeff, array_length);
	bool passed_qx1y  = compare_array_with_file_real(dirroot, filename_qx1y,  h_scale_coeff, array_length);
	bool passed_qy1y  = compare_array_with_file_real(dirroot, filename_qy1y,  h_scale_coeff, array_length);
	bool passed_z1y   = compare_array_with_file_real(dirroot, filename_z1y,   h_scale_coeff, array_length);

	bool passed =
	(
		passed_eta0  && passed_qx0  && passed_qy0  && passed_z0 &&
		passed_eta1x && passed_qx1x && passed_qy1x && passed_z1x &&
		passed_eta1y && passed_qx1y && passed_qy1y && passed_z1y
	);

	delete[] h_scale_coeff;

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
	real* h_scale_coeff    = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_scale_coeff[i] = i;
	}

	copy_cuda(d_scale_coeffs.eta0,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx0,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy0,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z0,    h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.eta1x, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx1x,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy1x,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z1x,   h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.eta1y, h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qx1y,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.qy1y,  h_scale_coeff, bytes);
	copy_cuda(d_scale_coeffs.z1y,   h_scale_coeff, bytes);
	
	const char* dirroot = "unittestdata";
	const char* prefix  = "unit_test_scale_coeffs_VERIFY_MW";

	const real actual_error   = d_scale_coeffs.verify(dirroot, prefix);
	const real expected_error = C(0.0);

	delete[] h_scale_coeff;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs_CONSTRUCTOR_COPY()
{
	SolverParams solver_params;

	solver_params.L           = 3;
	solver_params.solver_type = HWFV1;

	ScaleCoefficients d_scale_coeffs(solver_params);

	ScaleCoefficients d_scale_coeffs_copy(d_scale_coeffs);

	if (d_scale_coeffs.is_copy_cuda == false)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_classes()
{
	unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_HW();
	unit_test_scale_coeffs_CONSTRUCTOR_FILES_HW();
	unit_test_scale_coeffs_WRITE_TO_FILE_HW();
	unit_test_scale_coeffs_VERIFY_HW();

	unit_test_scale_coeffs_CONSTRUCTOR_LEVELS_MW();
	unit_test_scale_coeffs_CONSTRUCTOR_FILES_MW();
	unit_test_scale_coeffs_WRITE_TO_FILE_MW();
	unit_test_scale_coeffs_VERIFY_MW();

	unit_test_scale_coeffs_CONSTRUCTOR_COPY();
}

#endif