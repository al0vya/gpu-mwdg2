#include "compare_d_array_with_file_int.cuh"

int compare_d_array_with_file_int
(
	const char* dirroot,
	const char* filename,
	int*        d_array,
	const int&  array_length,
	const int&  offset
)
{
	int* h_array = new int[array_length];
	const size_t bytes = array_length * sizeof(int);
	
	copy_cuda(h_array, d_array, bytes);

	int differences = compare_array_with_file_int(dirroot, filename, h_array + offset, array_length - offset);

	delete[] h_array;

	return differences;
}