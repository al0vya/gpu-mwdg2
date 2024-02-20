#include "unit_tests_utilities.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_get_mean_from_array()
{
	const int array_length = 100000;
	const size_t bytes = array_length * sizeof(real);
	real* h_array = new real[array_length];
	real* d_array = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i+1;
	}

	copy_cuda(d_array, h_array, bytes);

	// sum S of 1 to n is S = n * (n+1) / 2
	// therefore, the mean M = S / n = (n+1)/2
	const real expected = (array_length + 1) / C(2.0);
	const real actual   = get_mean_from_array(d_array, array_length);

	delete[] h_array;
	free_device(d_array);

	if ( are_reals_equal( actual, expected, C(1e-2) ) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compute_error()
{
	const int array_length = 100000;
	const size_t bytes = array_length * sizeof(real);
	real* h_computed = new real[array_length];
	real* h_verified = new real[array_length];
	real* d_computed = (real*)malloc_device(bytes);
	real* d_verified = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_computed[i] =  i + 1;      // array is y = x
		h_verified[i] = (i + 1) * 2; // array is y = 2x
	}

	copy_cuda(d_computed, h_computed, bytes);
	copy_cuda(d_verified, h_verified, bytes);

	// array of errors = abs(d_computed - d_verified), which looks like abs(x - 2x) = x
	// i.e. the integers from 1 to 100000
	// sum S of integers 1 to n is S = n * (n+1) / 2
	// therefore, the mean M = S / n = (n+1)/2
	const real expected_error = (array_length + 1) / C(2.0);
	const real actual_error   = compute_error(d_computed, d_verified, array_length);

	delete[] h_computed;
	delete[] h_verified;
	free_device(d_computed);
	free_device(d_verified);

	if ( are_reals_equal( actual_error, expected_error, C(1e-2) ) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_utilities()
{
	unit_test_get_mean_from_array();
	unit_test_compute_error();
}

#endif