#include "read_test_case.h"

int read_test_case(const char* input_file)
{
	const int test_case = read_keyword_int(input_file, "test_case");

	// the number of synthetic test cases
	const int max_test_case_number = 23;

	if (test_case > max_test_case_number || test_case < 0)
	{
		printf("Error: please rerun and enter a number between 0 and %d for the test case number. Exiting program, file: %s, line: %d.\n", max_test_case_number, __FILE__, __LINE__);
		exit(-1);
	}

	return test_case;
}
