#include "get_mean_from_array.cuh"

__host__
real get_mean_from_array
(
	real*      d_array,
	const int& array_length
)
{
	
	real* h_sum_out = new real;
	real* d_sum_out = (real*)malloc_device( sizeof(real) );
	
	// Allocating memory to find maxes //

	void* d_temp_storage  = NULL;
	size_t  temp_storage  = 0;

	CHECK_CUDA_ERROR( cub::DeviceReduce::Sum
	(
		d_temp_storage,
		temp_storage,
		d_array,
		d_sum_out,
		array_length
	) );

	d_temp_storage = malloc_device(temp_storage);

	// ------------------------------- //

	CHECK_CUDA_ERROR( cub::DeviceReduce::Sum
	(
		d_temp_storage,
		temp_storage,
		d_array,
		d_sum_out,
		array_length
	) );

	copy_cuda
	(
		h_sum_out,
		d_sum_out,
		sizeof(real)
	);

	real mean_from_array = *h_sum_out / array_length;

	free_device(d_temp_storage);
	free_device(d_sum_out);
	delete      h_sum_out;

	return mean_from_array;
}