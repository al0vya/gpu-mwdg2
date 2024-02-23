#include "compare_d_array_with_file_bool.cuh"

bool compare_d_array_with_file_bool
(
	const char* dirroot,
	const char* filename,
	bool*       d_array,
	const int&  array_length
)
{
	bool* h_array = new bool[array_length];
	const size_t bytes = array_length * sizeof(bool);
	
	copy_cuda(h_array, d_array, bytes);

	bool passed = compare_array_with_file_bool(dirroot, filename, h_array, array_length);

	delete[] h_array;

	return passed;
}