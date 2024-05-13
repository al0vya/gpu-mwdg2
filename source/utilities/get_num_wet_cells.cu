#include "get_num_wet_cells.cuh"

__host__
real get_num_wet_cells
(
	real*       d_h0,
	const int&  array_length,
    const real& tol_h
)
{
	
	real* h_num_wet = new real;
	real* d_num_wet = (real*)malloc_device( sizeof(real) );
	
	// custom functor to detect and sum wet cells
	CountWet count_wet(tol_h);

	// --------------------//

	// Allocating memory to find maxes //

	void* d_temp_storage  = NULL;
	size_t  temp_storage  = 0;

	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_h0,
		d_num_wet,
		array_length,
		count_wet,
		C(0.0)
	) );

	d_temp_storage = malloc_device(temp_storage);

	// ------------------------------- //

	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_h0,
		d_num_wet,
		array_length,
		count_wet,
		C(0.0)
	) );

	copy_cuda
	(
		h_num_wet,
		d_num_wet,
		sizeof(real)
	);

	const real num_wet = *h_num_wet;

	free_device(d_temp_storage);
	free_device(d_num_wet);
	delete      h_num_wet;

	return num_wet;
}