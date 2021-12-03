#include "get_max_scale_coeffs.cuh"

__host__
Maxes get_max_scale_coeffs
(
	AssembledSolution& d_assem_sol,
	real*&             d_eta_temp
)
{
	// Variables for maxes //
	
	Maxes maxes = { C(1.0), C(1.0), C(1.0), C(1.0) };

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
		d_assem_sol.z0,
		d_max_out,
		d_assem_sol.length,
		abs_max,
		C(0.0)
	) );

	d_temp_storage = malloc_device(temp_storage);

	// ------------------------------- //

	// Finding maxes //

	// eta
	int num_blocks = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	
	init_eta_temp<<<num_blocks, THREADS_PER_BLOCK>>>
	(
		d_assem_sol, 
		d_eta_temp
	);

	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_eta_temp,
		d_max_out,
		d_assem_sol.length,
		abs_max,
		C(0.0)
	) );

	copy
	(
		h_max_out,
		d_max_out,
		sizeof(real)
	);

	maxes.eta = max( *h_max_out, C(1.0) );

	// qx
	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_assem_sol.qx0,
		d_max_out,
		d_assem_sol.length,
		abs_max,
		C(0.0)
	) );

	copy
	(
		h_max_out,
		d_max_out,
		sizeof(real)
	);

	maxes.qx = max( *h_max_out, C(1.0) );

	// qy
	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_assem_sol.qy0,
		d_max_out,
		d_assem_sol.length,
		abs_max,
		C(0.0)
	) );

	copy
	(
		h_max_out,
		d_max_out,
		sizeof(real)
	);

	maxes.qy = max( *h_max_out, C(1.0) );

	// z
	CHECK_CUDA_ERROR( cub::DeviceReduce::Reduce
	(
		d_temp_storage,
		temp_storage,
		d_assem_sol.z0,
		d_max_out,
		d_assem_sol.length,
		abs_max,
		C(0.0)
	) );

	copy
	(
		h_max_out,
		d_max_out,
		sizeof(real)
	);

	maxes.z = max( *h_max_out, C(1.0) );

	// ------------- //

	free_device(d_max_out);
	free_device(d_temp_storage);
	delete h_max_out;

	return maxes;
}