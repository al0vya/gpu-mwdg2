#include "compute_error.cuh"

real compute_error
(
	const char* dirroot,
	const char* filename,
	real*       d_computed,
	real*       d_verified,
	const int&  array_length
)
{
	const int num_blocks = get_num_blocks(array_length, THREADS_PER_BLOCK);

	compute_error_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_computed, d_verified, array_length);

	real mean_error = get_mean_from_array(d_verified, array_length);

	return mean_error;
}