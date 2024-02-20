#include "get_max_from_array.cuh"

__host__
real get_max_from_array
(
	real*      d_array,
	const int& array_length
)
{
	
	real* h_max_out = new real;
	real* d_max_out = (real*)malloc_device( sizeof(real) );
	
	// custom functor for maximum of absolute value
	AbsMax abs_max;

	// --------------------//

	// Allocating memory to find maxes //

	void* d_temp_storage  = NULL;
	size_t  temp_storage  = 0;

	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_array,
		d_max_out,
		array_length,
		abs_max,
		C(0.0)
	) );

	d_temp_storage = malloc_device(temp_storage);

	// ------------------------------- //

	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_array,
		d_max_out,
		array_length,
		abs_max,
		C(0.0)
	) );

	copy_cuda
	(
		h_max_out,
		d_max_out,
		sizeof(real)
	);

	real max_from_array = max( *h_max_out, C(1.0) );

	free_device(d_temp_storage);
	free_device(d_max_out);
	delete      h_max_out;

	return max_from_array;
}