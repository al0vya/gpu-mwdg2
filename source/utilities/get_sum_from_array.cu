#include "get_sum_from_array.cuh"

__host__
int get_sum_from_array
(
	bool*      d_array,
	const int& array_length
)
{
	
	int* h_sum_out = new int;
	int* d_sum_out = (int*)malloc_device( sizeof(int) );
	
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
		sizeof(int)
	);

	int sum_from_array = *h_sum_out;

	free_device(d_temp_storage);
	free_device(d_sum_out);
	delete      h_sum_out;

	return sum_from_array;
}