#include "unit_tests_utilities.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_get_max_from_array()
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

	const real expected = array_length;
	const real actual   = get_max_from_array(d_array, array_length);

	delete[] h_array;
	free_device(d_array);

	if ( are_reals_equal( actual, expected, C(1e-2) ) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

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

void unit_test_compute_max_error()
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
	// i.e. the integers from 1 to 100000, therefore max error is 100000, i.e. array_length
	const real expected_error = array_length;
	const real actual_error   = compute_max_error(d_computed, d_verified, array_length);

	delete[] h_computed;
	delete[] h_verified;
	free_device(d_computed);
	free_device(d_verified);

	if ( are_reals_equal( actual_error, expected_error, C(1e-2) ) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_array_on_device_vs_host_int()
{
	const int array_length = 100;
	const size_t bytes = array_length * sizeof(int);
	int* h_array = new int[array_length];
	int* d_array = (int*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i % 2;
	}

	copy_cuda(d_array, h_array, bytes);

	const int diffs = compare_array_on_device_vs_host_int(h_array, d_array, array_length);
	
	delete[] h_array;
	free_device(d_array);

	if (diffs == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_array_on_device_vs_host_real()
{
	const int array_length = 100;
	const size_t bytes = array_length * sizeof(real);
	real* h_array = new real[array_length];
	real* d_array = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i;
	}

	copy_cuda(d_array, h_array, bytes);

	const real actual_error   = compare_array_on_device_vs_host_real(h_array, d_array, array_length);
	const real expected_error = C(1e-6);

	delete[] h_array;
	free_device(d_array);

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_array_with_file_bool()
{
	const int array_length = 100;
	bool* h_array = new bool[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i % 2 == 0;
	}

	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_array_with_file_bool";

	const int differences = compare_array_with_file_bool(dirroot, filename, h_array, array_length);

	delete[] h_array;

	if (differences == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_array_with_file_int()
{
	const int array_length = 100;
	int* h_array = new int[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i % 2 == 0;
	}

	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_array_with_file_int";

	const int differences = compare_array_with_file_int(dirroot, filename, h_array, array_length);

	delete[] h_array;

	if (differences == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_array_with_file_real()
{
	const int array_length = 100;
	real* h_array = new real[array_length];

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i;
	}

	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_array_with_file_real";

	const real actual_error   = compare_array_with_file_real(dirroot, filename, h_array, array_length);
	const real expected_error = C(1e-6);

	delete[] h_array;

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_d_array_with_file_bool_NO_OFFSET()
{
	const int array_length = 100;
	const size_t bytes = array_length * sizeof(bool);
	bool* h_array = new bool[array_length];
	bool* d_array = (bool*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i % 2 == 0;
	}
	
	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_d_array_with_file_bool_NO_OFFSET";

	copy_cuda(d_array, h_array, bytes);

	const int differences = compare_d_array_with_file_bool(dirroot, filename, d_array, array_length, 0);

	delete[] h_array;
	free_device(d_array);

	if (differences == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_d_array_with_file_bool_OFFSET()
{
	const int offset = 50;
	const int array_length = 100;
	const int array_length_offset = array_length + offset;
	const size_t bytes = array_length_offset * sizeof(bool);
	bool* h_array = new bool[array_length_offset];
	bool* d_array = (bool*)malloc_device(bytes);

	for (int i = 0; i < array_length_offset; i++)
	{
		h_array[i] = (i < offset) ? 0 : (i - offset) % 2 == 0;
	}
	
	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_d_array_with_file_bool_OFFSET";

	copy_cuda(d_array, h_array, bytes);

	const int differences = compare_d_array_with_file_bool(dirroot, filename, d_array, array_length_offset, offset);

	delete[] h_array;
	free_device(d_array);

	if (differences == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_d_array_with_file_int_NO_OFFSET()
{
	const int array_length = 100;
	const size_t bytes = array_length * sizeof(int);
	int* h_array = new int[array_length];
	int* d_array = (int*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i % 2 == 0;
	}
	
	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_d_array_with_file_int_NO_OFFSET";

	copy_cuda(d_array, h_array, bytes);

	const int differences = compare_d_array_with_file_int(dirroot, filename, d_array, array_length, 0);

	delete[] h_array;
	free_device(d_array);

	if (differences == 0)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_d_array_with_file_real_NO_OFFSET()
{
	const int array_length = 100;
	const size_t bytes = array_length * sizeof(real);
	real* h_array = new real[array_length];
	real* d_array = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_array[i] = i;
	}
	
	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_d_array_with_file_real_NO_OFFSET";

	copy_cuda(d_array, h_array, bytes);

	const real actual_error   = compare_d_array_with_file_real(dirroot, filename, d_array, array_length, 0);
	const real expected_error = C(1e-6);

	delete[] h_array;
	free_device(d_array);

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_compare_d_array_with_file_real_OFFSET()
{
	const int offset = 50;
	const int array_length = 100;
	const int array_length_offset = array_length + offset;
	const size_t bytes = array_length_offset * sizeof(real);
	real* h_array = new real[array_length_offset];
	real* d_array = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length_offset; i++)
	{
		h_array[i] = (i < offset) ? C(0.0) : i - offset;
	}
	
	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_compare_d_array_with_file_real_OFFSET";

	copy_cuda(d_array, h_array, bytes);

	const real actual_error   = compare_d_array_with_file_real(dirroot, filename, d_array, array_length_offset, offset);
	const real expected_error = C(1e-6);

	delete[] h_array;
	free_device(d_array);

	if ( are_reals_equal(actual_error, expected_error) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_zero_array_kernel_int()
{
	const int array_length = 100000;
	const int num_blocks = get_num_blocks(array_length, THREADS_PER_BLOCK);
	const size_t bytes = array_length * sizeof(int);
	int* h_array = new int[array_length];
	int* d_array = (int*)malloc_device(bytes);

	zero_array_kernel_int<<<num_blocks, THREADS_PER_BLOCK>>>(d_array, array_length);

	copy_cuda(h_array, d_array, bytes);

	bool passed = true;

	for (int i = 0; i < array_length; i++)
	{
		if ( h_array[i] != 0 )
		{
			passed = false;
			break;
		}
	}

	delete[] h_array;
	free_device(d_array);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_zero_array_kernel_real()
{
	const int array_length = 100000;
	const int num_blocks = get_num_blocks(array_length, THREADS_PER_BLOCK);
	const size_t bytes = array_length * sizeof(real);
	real* h_array = new real[array_length];
	real* d_array = (real*)malloc_device(bytes);

	zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(d_array, array_length);

	copy_cuda(h_array, d_array, bytes);

	bool passed = true;

	for (int i = 0; i < array_length; i++)
	{
		if ( h_array[i] != C(0.0) )
		{
			passed = false;
			break;
		}
	}

	delete[] h_array;
	free_device(d_array);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_utilities()
{
	unit_test_get_max_from_array();
	unit_test_get_mean_from_array();
	unit_test_compute_max_error();
	unit_test_compare_array_on_device_vs_host_int();
	unit_test_compare_array_on_device_vs_host_real();
	unit_test_compare_array_with_file_bool();
	unit_test_compare_array_with_file_int();
	unit_test_compare_array_with_file_real();
	unit_test_compare_d_array_with_file_bool_NO_OFFSET();
	unit_test_compare_d_array_with_file_bool_OFFSET();
	unit_test_compare_d_array_with_file_int_NO_OFFSET();
	unit_test_compare_d_array_with_file_real_NO_OFFSET();
	unit_test_compare_d_array_with_file_real_OFFSET();
	unit_test_zero_array_kernel_int();
	unit_test_zero_array_kernel_real();
}

#endif