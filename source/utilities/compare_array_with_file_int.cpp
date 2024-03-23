#include "compare_array_with_file_int.h"

int compare_array_with_file_int
(
	const char* dirroot,
	const char* filename,
	int*        h_array,
	const int&  array_length
)
{
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s for int comparison.\n", fullpath);
		return 999.0;
	}

	int diffs      = 0;
	int host_value = 0;
	int file_value = 0;

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];
		
		fscanf(fp, "%d", &file_value);

		diffs += (host_value != file_value);
	}

	fclose(fp);

	return diffs;
}