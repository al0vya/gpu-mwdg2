#include "compare_array_with_file_real.h"

real compare_array_with_file_real
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
		return C(999.0);
	}

	real error      = C(0.0);
	real max_error  = C(0.0);
	real host_value = C(0.0);
	real file_value = C(0.0);

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];
		
		fscanf(fp, "%f", &file_value);

		error = std::abs(host_value - file_value);

		max_error = std::max(max_error, error);
	}

	fclose(fp);

	return max_error;
}