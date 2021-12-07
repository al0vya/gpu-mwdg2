#include "rev_z_order_reals.cuh"

__host__
void rev_z_order_reals
(
	MortonCode* d_rev_z_order,
	MortonCode* d_indices,
	real*       d_array,
	real*       d_array_sorted,
	int         array_length
)
{
	void*  d_temp_storage = NULL;
	size_t temp_storage   = 0;

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_array,
		d_array_sorted,
		array_length
	) );

	d_temp_storage = malloc_device(temp_storage);

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_array,
		d_array_sorted,
		array_length
	) );

	free_device(d_temp_storage);
}