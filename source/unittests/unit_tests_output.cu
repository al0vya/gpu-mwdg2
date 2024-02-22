#include "unit_tests_output.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_write_hierarchy_array_real()
{
	const int    levels       = 3;
	const int    array_length = get_lvl_idx(levels + 1);
	      size_t bytes        = array_length * sizeof(real);
	      real*  h_hierarchy  = new real[array_length];
	      real*  d_hierarchy  = (real*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_hierarchy[i] = i;
	}

	copy_cuda(d_hierarchy, h_hierarchy, bytes);

	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_write_hierarchy_array_real";

	write_hierarchy_array_real(dirroot, filename, d_hierarchy, levels);

	bool passed = compare_array_with_file_real(dirroot, filename, h_hierarchy, array_length);

	delete[]    h_hierarchy;
	free_device(d_hierarchy);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_write_hierarchy_array_bool()
{
	const int    levels       = 3;
	const int    array_length = get_lvl_idx(levels + 1);
	      size_t bytes        = array_length * sizeof(real);
	      bool*  h_hierarchy  = new bool[array_length];
	      bool*  d_hierarchy  = (bool*)malloc_device(bytes);

	for (int i = 0; i < array_length; i++)
	{
		h_hierarchy[i] = (i % 2 == 0);
	}

	copy_cuda(d_hierarchy, h_hierarchy, bytes);

	const char* dirroot  = "unittestdata";
	const char* filename = "unit_test_write_hierarchy_array_bool";

	write_hierarchy_array_bool(dirroot, filename, d_hierarchy, levels);

	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s, failed %s\n.", fullpath, __func__);
		return;
	}

	bool passed = true;

	int host_value = 0;
	int file_value = 0;

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_hierarchy[i];
		
		fscanf(fp, "%d", &file_value);

		if (host_value != file_value)
		{
			passed = false;
			break;
		}
	}

	fclose(fp);

	delete[]    h_hierarchy;
	free_device(d_hierarchy);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void run_unit_tests_output()
{
	unit_test_write_hierarchy_array_real();
	unit_test_write_hierarchy_array_bool();
}

#endif