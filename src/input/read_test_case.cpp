#include "read_test_case.h"

int read_test_case(const char* input_file)
{
	FILE* fp = fopen(input_file, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening input file for test case, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	char str[255];
	char buf[64];
	int  test_case = 0;

	while ( strncmp(buf, "test_case", 9) )
	{
		if (NULL == fgets(str, sizeof(str), fp))
		{
			fprintf(stderr, "Error reading input file for test case, file: %s, line: %d.\n", __FILE__, __LINE__);
			fclose(fp);
			exit(-1);
		}

		int num_read = sscanf(str, "%s %d", buf, &test_case);
	}

	fclose(fp);
	
	printf("Test case: %d\n", test_case);

	const int max_test_case = 22;

	if (test_case > max_test_case || test_case < 0)
	{
		printf("Error: please rerun and enter a number between 0 and %d. Exiting program, file: %s, line: %d.\n", max_test_case, __FILE__, __LINE__);
		exit(-1);
	}

	return test_case;
}