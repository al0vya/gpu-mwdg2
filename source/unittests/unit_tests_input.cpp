#include "unit_tests_input.h"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed unit test %s!\n", __func__); } else { printf("Failed unit test %s.\n", __func__); }

bool are_reals_equal
(
	const real& a,
	const real& b,
	const real& epsilon
)
{
	return fabs(a - b) <= epsilon;
}

void test_read_keyword_int_KEYWORD_NOT_FOUND()
{
	// file looks like:
	// dummy 1
	const char* filename = "unit_test_read_keyword_int_KEYWORD_NOT_FOUND.txt";

	const char* keyword = "keyword";

	const int expected = 0; // read_keyword_int should return 0 if keyword not found

	const int actual = read_keyword_int(filename, keyword);

	if (actual == expected)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_read_keyword_int_KEYWORD_FOUND()
{
	// file looks like:
	// keyword 1
	const char* filename = "unit_test_read_keyword_int_KEYWORD_FOUND.txt";

	const char* keyword = "keyword";

	const int expected = 1;

	const int actual = read_keyword_int(filename, keyword);

	if (actual == expected)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_read_keyword_str_KEYWORD_NOT_FOUND()
{
	// file looks like:
	// #DEMfile monai.dem
	const char* filename = "unit_test_read_keyword_str_KEYWORD_NOT_FOUND.txt";

	const char* keyword = "DEMfile";

	char value_buf[128];

	read_keyword_str(filename, keyword, value_buf);

	// read_keyword_str should set first char of value_buf to '\0' if keyword not found
	if ( value_buf[0] == '\0')
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_read_keyword_str_KEYWORD_FOUND()
{
	// file looks like:
	// DEMfile monai.dem
	const char* filename = "unit_test_read_keyword_str_KEYWORD_FOUND.txt";

	const char* keyword = "DEMfile";

	const char* expected = "monai.dem";

	const int num_char_expected = strlen(expected);

	char value_buf[128] = {'\0'};

	read_keyword_str(filename, keyword, value_buf);

	if ( !strncmp(expected, value_buf, num_char_expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_read_cell_size_CELL_SIZE_FOUND()
{
	// file looks like:
	// ncols        392
	// nrows        243
	// xllcorner    0
	// yllcorner    0
	// cellsize     0.014
	// NODATA_value -9999
	// DEMfile      unit_test_read_cell_size_CELL_SIZE_FOUND.txt
	const char* input_filename = "unit_test_read_cell_size_CELL_SIZE_FOUND.txt";

	const real expected = C(0.014);

	const real actual = read_cell_size(input_filename);

	if ( are_reals_equal( expected, actual, C(1e-10) ) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_input()
{
	test_read_keyword_int_KEYWORD_NOT_FOUND();
	test_read_keyword_int_KEYWORD_FOUND();
	test_read_keyword_str_KEYWORD_NOT_FOUND();
	test_read_keyword_str_KEYWORD_FOUND();
	test_read_cell_size_CELL_SIZE_FOUND();
}

#endif