#include "unit_tests_input.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_read_keyword_int_KEYWORD_NOT_FOUND()
{
	// file looks like:
	// dummy 1
	const char* filename = "unittestdata/unit_test_read_keyword_int_KEYWORD_NOT_FOUND.txt";

	const char* keyword = "keyword";

	const int expected = 0; // read_keyword_int should return 0 if keyword not found

	const int actual = read_keyword_int(filename, keyword);

	if (actual == expected)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_keyword_int_KEYWORD_FOUND()
{
	// file looks like:
	// keyword 1
	const char* filename = "unittestdata/unit_test_read_keyword_int_KEYWORD_FOUND.txt";

	const char* keyword = "keyword";

	const int expected = 1;

	const int actual = read_keyword_int(filename, keyword);

	if (actual == expected)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_keyword_str_KEYWORD_NOT_FOUND()
{
	// file looks like:
	// #DEMfile monai.dem
	const char* filename = "unittestdata/unit_test_read_keyword_str_KEYWORD_NOT_FOUND.txt";

	const char* keyword = "DEMfile";

	char value_buf[128];

	read_keyword_str(filename, keyword, value_buf);

	// read_keyword_str should set first char of value_buf to '\0' if keyword not found
	if ( value_buf[0] == '\0')
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_keyword_str_KEYWORD_FOUND()
{
	// file looks like:
	// DEMfile monai.dem
	const char* filename = "unittestdata/unit_test_read_keyword_str_KEYWORD_FOUND.txt";

	const char* keyword = "DEMfile";

	const char* expected = "monai.dem";

	const int num_char_expected = strlen(expected);

	char value_buf[128] = {'\0'};

	read_keyword_str(filename, keyword, value_buf);

	if ( !strncmp(expected, value_buf, num_char_expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_cell_size_CELL_SIZE_FOUND()
{
	// file looks like:
	// ncols        392
	// nrows        243
	// xllcorner    0
	// yllcorner    0
	// cellsize     0.014
	// NODATA_value -9999
	// DEMfile      unittestdata/unit_test_read_cell_size_CELL_SIZE_FOUND.txt
	const char* input_filename = "unittestdata/unit_test_read_cell_size_CELL_SIZE_FOUND.txt";

	const real expected = C(0.014);

	const real actual = read_cell_size(input_filename);

	if ( are_reals_equal(expected, actual) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_d_array_int()
{
	const char*  dirroot      = "unittestdata";
	const char*  filename     = "unit_test_read_d_array_int";
	const int    array_length = 10;
	const size_t bytes        = array_length * sizeof(int);

	int* d_hierarchy = read_d_array_int(array_length, dirroot, filename);
	int* h_hierarchy = new int[array_length];

	copy_cuda(h_hierarchy, d_hierarchy, bytes);

	bool passed = true;

	for (int i = 0; i < array_length; i++)
	{
		if ( i != h_hierarchy[i] )
		{
			passed = false;
			break;
		}
	}

	free_device(d_hierarchy);
	delete[]    h_hierarchy;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_d_array_real()
{
	const char*  dirroot      = "unittestdata";
	const char*  filename     = "unit_test_read_d_array_real";
	const int    array_length = 10;
	const size_t bytes        = array_length * sizeof(real);

	real* d_hierarchy = read_d_array_real(array_length, dirroot, filename);
	real* h_hierarchy = new real[array_length];

	copy_cuda(h_hierarchy, d_hierarchy, bytes);

	bool passed = true;

	real dummy = C(0.0);

	for (int i = 0; i < array_length; i++)
	{
		dummy = i;
		
		if ( !are_reals_equal( dummy, h_hierarchy[i] ) )
		{
			passed = false;
			break;
		}
	}

	free_device(d_hierarchy);
	delete[]    h_hierarchy;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_hierarchy_array_real()
{
	const char*  dirroot      = "unittestdata";
	const char*  filename     = "unit_test_read_hierarchy_array_real";
	const int    levels       = 3;
	const int    array_length = get_lvl_idx(levels + 1);
	const size_t bytes        = array_length * sizeof(real);

	real* d_hierarchy = read_hierarchy_array_real(levels, dirroot, filename);
	real* h_hierarchy = new real[array_length];

	copy_cuda(h_hierarchy, d_hierarchy, bytes);

	bool passed = true;

	real dummy = C(0.0);

	for (int i = 0; i < array_length; i++)
	{
		dummy = (i < PADDING_MRA) ? C(0.0) : i - PADDING_MRA; // to get a series of numbers with the first three being 0: 0, 0, 0, 0, 1, 2, ..
		
		if ( !are_reals_equal( dummy, h_hierarchy[i] ) )
		{
			passed = false;
			break;
		}
	}

	free_device(d_hierarchy);
	delete[]    h_hierarchy;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_read_hierarchy_array_bool()
{
	const char*  dirroot      = "unittestdata";
	const char*  filename     = "unit_test_read_hierarchy_array_bool";
	const int    levels       = 3;
	const int    array_length = get_lvl_idx(levels + 1);
	const size_t bytes        = array_length * sizeof(bool);

	bool* d_hierarchy = read_hierarchy_array_bool(levels, dirroot, filename);
	bool* h_hierarchy = new bool[array_length];

	copy_cuda(h_hierarchy, d_hierarchy, bytes);

	bool passed = true;

	bool dummy = 0;

	for (int i = 0; i < array_length; i++)
	{
		dummy = (i < PADDING_MRA) ? 0 : ( (i - PADDING_MRA) % 2 == 0 );
		
		if ( dummy != h_hierarchy[i] )
		{
			passed = false;
			break;
		}
	}

	free_device(d_hierarchy);
	delete[]    h_hierarchy;

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_input()
{
	unit_test_read_keyword_int_KEYWORD_NOT_FOUND();
	unit_test_read_keyword_int_KEYWORD_FOUND();
	unit_test_read_keyword_str_KEYWORD_NOT_FOUND();
	unit_test_read_keyword_str_KEYWORD_FOUND();
	unit_test_read_cell_size_CELL_SIZE_FOUND();
	unit_test_read_d_array_int();
	unit_test_read_d_array_real();
	unit_test_read_hierarchy_array_real();
	unit_test_read_hierarchy_array_bool();
}

#endif