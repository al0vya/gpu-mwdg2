#include "sort_finest_scale_coeffs_z_order.cuh"

__host__
void sort_finest_scale_coeffs_z_order
(
	MortonCode*        d_morton_codes,
	MortonCode*        d_sorted_morton_codes,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	MortonCode*        d_indices,
	MortonCode*        d_rev_z_order,
	SolverParameters&  solver_params
)
{
	// ------------------------------ //
	// Sorting the scale coefficients //
	// ------------------------------ //
	
	void* d_temp_storage = NULL;
	size_t temp_storage  = 0;

	// this launch only decides how much temp_storage is needed for allocation to d_temp_storage
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_assem_sol.z0, 
		d_buf_assem_sol.z0, 
		d_assem_sol.length
	) );

	d_temp_storage = malloc_device(temp_storage);

	// sorting the Morton codes is equivalent to ordering the scale coefficients according to
	// a z-order curve, please see: https://en.wikipedia.org/wiki/Z-order_curve
	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_assem_sol.h0, 
		d_buf_assem_sol.h0, 
		d_assem_sol.length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_assem_sol.qx0, 
		d_buf_assem_sol.qx0, 
		d_assem_sol.length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_assem_sol.qy0, 
		d_buf_assem_sol.qy0, 
		d_assem_sol.length
	) );

	CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
	(
		d_temp_storage,
		temp_storage,
		d_morton_codes,
		d_sorted_morton_codes, 
		d_assem_sol.z0, 
		d_buf_assem_sol.z0, 
		d_assem_sol.length
	) );

	if (solver_params.solver_type == MWDG2)
	{
		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.h1x,
			d_buf_assem_sol.h1x,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.qx1x,
			d_buf_assem_sol.qx1x,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.qy1x,
			d_buf_assem_sol.qy1x,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.z1x,
			d_buf_assem_sol.z1x,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.h1y,
			d_buf_assem_sol.h1y,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.qx1y,
			d_buf_assem_sol.qx1y,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.qy1y,
			d_buf_assem_sol.qy1y,
			d_assem_sol.length
		) );

		CHECK_CUDA_ERROR( cub::DeviceRadixSort::SortPairs
		(
			d_temp_storage,
			temp_storage,
			d_morton_codes,
			d_sorted_morton_codes,
			d_assem_sol.z1y,
			d_buf_assem_sol.z1y,
			d_assem_sol.length
		) );
	}

	free_device(d_temp_storage);

	// ------------------------------ //

	// ---------------------------------------------- //
	// Getting array with which to reverse z-ordering //
	// ---------------------------------------------- //

	d_temp_storage = NULL;
	temp_storage   = 0;

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

	free_device(d_temp_storage);

	// ---------------------------------------------- //
}