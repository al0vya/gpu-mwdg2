#include "../zorder/rev_z_order_assem_sol.cuh"

void rev_z_order_assem_sol
(
	MortonCode*       d_rev_z_order,
	MortonCode*       d_indices,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	int               array_length
)
{
	void*  d_temp_storage = NULL;
	size_t temp_storage   = 0;

	// this launch only decides how much temp_storage is needed for allocation to d_temp_storage
	// use the large possible data type, here real in d_assem_sol.z, to allocate a largest d_temp_storage
	// that will be able to accommodate all further sorts
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.z0,
		d_assem_sol.z0,
		array_length
	) );

	d_temp_storage = malloc_device(temp_storage);

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.h0,
		d_assem_sol.h0,
		array_length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.qx0,
		d_assem_sol.qx0,
		array_length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.qy0,
		d_assem_sol.qy0,
		array_length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.z0,
		d_assem_sol.z0,
		array_length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.levels,
		d_assem_sol.levels,
		array_length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol.act_idcs,
		d_assem_sol.act_idcs,
		array_length
	) );

	free_device(d_temp_storage);
}