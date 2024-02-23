#include "compare_array_with_file_bool.h"

bool compare_array_with_file_bool
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

	bool passed = true;

	int host_value = 0;
	int file_value = 0;

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];
		
		fscanf(fp, "%d", &file_value);

		if (host_value != file_value)
		{
			passed = false;
			break;
		}
	}

	fclose(fp);

	return passed;
}