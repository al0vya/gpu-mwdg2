#include "compare_d_array_with_file_real.cuh"

bool compare_d_array_with_file_real
(
	const char* dirroot,
	const char* filename,
	real*       d_array,
	const int&  array_length
)
{
	real* h_array = new real[array_length];
	const size_t bytes = array_length * sizeof(real);
	
	copy_cuda(h_array, d_array, bytes);

	bool passed = compare_array_with_file_real(dirroot, filename, h_array, array_length);

	delete[] h_array;

	return passed;
}