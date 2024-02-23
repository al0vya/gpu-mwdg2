#include "compute_error.cuh"

real compute_error
(
	real*       d_computed,
	real*       d_verified,
	const int&  array_length
)
{
	const size_t bytes = array_length * sizeof(real);

	real* d_errors = (real*)malloc_device(bytes);
	
	const int num_blocks = get_num_blocks(array_length, THREADS_PER_BLOCK);

	compute_error_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
	(
		d_computed,
		d_verified,
		d_errors,
		array_length
	);

	real max_error = get_max_from_array(d_errors, array_length);

	free_device(d_errors);

	return max_error;
}