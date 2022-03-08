#include "get_sorting_indices.cuh"

__host__
void get_sorting_indices
(
	MortonCode*        d_morton_codes,
	MortonCode*        d_sorted_morton_codes,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	MortonCode*        d_indices,
	MortonCode*        d_rev_z_order,
	MortonCode*        d_rev_row_major,
	SolverParams&      solver_params
)
{
	// ---------------------------------------------- //
	// Getting array with which to reverse z-ordering //
	// ---------------------------------------------- //

	void*  d_temp_storage = NULL;
	size_t temp_storage   = 0;

	// this launch only decides how much temp_storage is needed for allocation to d_temp_storage
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_indices, 
		d_rev_z_order,
		d_assem_sol.length
	) );

	d_temp_storage = malloc_device(temp_storage);

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_indices,
		d_rev_z_order,
		d_assem_sol.length
	) );
	
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_rev_z_order,
		d_sorted_morton_codes, 
		d_indices,
		d_rev_row_major,
		d_assem_sol.length
	) );

	free_device(d_temp_storage);

	// ---------------------------------------------- //
}