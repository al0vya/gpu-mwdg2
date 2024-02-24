#include "compare_array_with_file_bool.h"

int compare_array_with_file_bool
(
	const char* dirroot,
	const char* filename,
	bool*       h_array,
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

	int differences = 0;
	int host_value  = 0;
	int file_value  = 0;

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];
		
		fscanf(fp, "%d", &file_value);

		differences += (host_value - file_value) != 0;
	}

	fclose(fp);

	return differences;
}