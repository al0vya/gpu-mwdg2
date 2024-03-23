#include "compare_array_on_device_vs_host_int.cuh"

int compare_array_on_device_vs_host_int
(
	int*       h_array,
	int*       d_array,
	const int& array_length
)
{
	const size_t bytes = array_length * sizeof(int);
	int* h_array_copied = new int[array_length];

	copy_cuda(h_array_copied, d_array, bytes);

	int diffs      = 0;
	int host_value = 0;
	int file_value = 0;

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];

		file_value = h_array_copied[i];

		diffs += (host_value != file_value);
	}

	delete[] h_array_copied;

	return diffs;
}