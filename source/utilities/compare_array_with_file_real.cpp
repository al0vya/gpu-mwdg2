#include "compare_array_with_file_real.h"

bool compare_array_with_file_real
(
	const char* dirroot,
	const char* filename,
	real*       h_array,
	const int&  array_length
)
{
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s\n.", fullpath);
		return false;
	}

	bool passed = true;

	real host_value = C(0.0);
	real file_value = C(0.0);

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];
		
		fscanf(fp, "%f", &file_value);

		if ( !are_reals_equal(host_value, file_value) )
		{
			passed = false;
			break;
		}
	}

	fclose(fp);

	return passed;
}