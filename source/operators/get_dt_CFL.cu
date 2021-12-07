#include "get_dt_CFL.cuh"

__host__
real get_dt_CFL
(
	real*&     d_dt_CFL,
	const int& sol_len
)
{
	real* h_min_out = new real;
	real* d_min_out = (real*)malloc_device(sizeof(real));

	void*  d_temp_storage = NULL;
	size_t temp_storage  = 0;

	CHECK_CUDA_ERROR( cub::DeviceReduce::Min
	(
		d_temp_storage,
		temp_storage,
		d_dt_CFL,
		d_min_out,
		sol_len
	) );

	d_temp_storage = malloc_device(temp_storage);

	CHECK_CUDA_ERROR( cub::DeviceReduce::Min
	(
		d_temp_storage,
		temp_storage,
		d_dt_CFL,
		d_min_out,
		sol_len
	) );

	copy
	(
		h_min_out, 
		d_min_out, 
		sizeof(real)
	);

	real dt_min = *h_min_out;

	free_device(d_min_out);
	free_device(d_temp_storage);
	delete h_min_out;

	return dt_min;
}