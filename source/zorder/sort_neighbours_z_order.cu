#include "sort_neighbours_z_order.cuh"

__host__
void sort_neighbours_z_order
(
	const Neighbours&  d_neighbours,
	const Neighbours&  d_buf_neighbours,
	MortonCode*        d_morton_codes,
	MortonCode*        d_sorted_morton_codes,
	int                num_finest_elems,
	const SolverParams& solver_params
)
{
	void*  d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	
	// first call only returns the amount of memory needed for d_temp_storage, in temp_storage_bytes
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_morton_codes, 
		d_sorted_morton_codes, 
		d_neighbours.north.act_idcs, 
		d_buf_neighbours.north.act_idcs,
		num_finest_elems
	) );

	d_temp_storage = malloc_device(temp_storage_bytes);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_morton_codes, 
		d_sorted_morton_codes, 
		d_neighbours.north.act_idcs, 
		d_buf_neighbours.north.act_idcs,
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_morton_codes, 
		d_sorted_morton_codes, 
		d_neighbours.east.act_idcs, 
		d_buf_neighbours.east.act_idcs,
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_morton_codes, 
		d_sorted_morton_codes, 
		d_neighbours.south.act_idcs, 
		d_buf_neighbours.south.act_idcs,
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_morton_codes, 
		d_sorted_morton_codes, 
		d_neighbours.west.act_idcs, 
		d_buf_neighbours.west.act_idcs,
		num_finest_elems
	) );

	if (solver_params.solver_type == MWDG2)
	{
		CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage_bytes,
			d_morton_codes,
			d_sorted_morton_codes,
			d_neighbours.north.levels,
			d_buf_neighbours.north.levels,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage_bytes,
			d_morton_codes,
			d_sorted_morton_codes,
			d_neighbours.east.levels,
			d_buf_neighbours.east.levels,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage_bytes,
			d_morton_codes,
			d_sorted_morton_codes,
			d_neighbours.south.levels,
			d_buf_neighbours.south.levels,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage_bytes,
			d_morton_codes,
			d_sorted_morton_codes,
			d_neighbours.west.levels,
			d_buf_neighbours.west.levels,
			num_finest_elems
		));
	}

	CHECK_CUDA_ERROR( free_device(d_temp_storage) );
}