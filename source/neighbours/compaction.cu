#include "compaction.cuh"

__host__
void compaction
(
	AssembledSolution& d_buf_assem_sol, 
	AssembledSolution& d_assem_sol, 
	Neighbours&        d_buf_neighbours, 
	Neighbours&        d_neighbours, 
	CompactionFlags&   d_compaction_flags,
	int                num_finest_elems,
	const SolverParams& solver_params
)
{
	void*  d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;

	int* d_assem_sol_len = (int*)malloc_device( sizeof(int) );
	int* h_sol_len = new int;

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.z0, 
		d_compaction_flags.north_east, 
		d_assem_sol.z0, 
		d_assem_sol_len, 
		num_finest_elems
	) );

	d_temp_storage = malloc_device(temp_storage_bytes);

	/*CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.h0, 
		d_compaction_flags.north_east, 
		d_assem_sol.h0, 
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.qx0, 
		d_compaction_flags.north_east, 
		d_assem_sol.qx0, 
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.qy0, 
		d_compaction_flags.north_east, 
		d_assem_sol.qy0, 
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.z0, 
		d_compaction_flags.north_east, 
		d_assem_sol.z0, 
		d_assem_sol_len, 
		num_finest_elems
	) );*/

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.act_idcs, 
		d_compaction_flags.north_east, 
		d_assem_sol.act_idcs, 
		d_assem_sol_len, 
		num_finest_elems
	) );

	copy_cuda
	(
		h_sol_len, 
		d_assem_sol_len, 
		sizeof(int)
	);
	
	d_assem_sol.length = *h_sol_len;
	
	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_assem_sol.levels, 
		d_compaction_flags.north_east, 
		d_assem_sol.levels,
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_neighbours.north.act_idcs, 
		d_compaction_flags.north_east, 
		d_neighbours.north.act_idcs,
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_neighbours.east.act_idcs,
		d_compaction_flags.north_east, 
		d_neighbours.east.act_idcs,
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_neighbours.south.act_idcs,
		d_compaction_flags.south_west, 
		d_neighbours.south.act_idcs,
		d_assem_sol_len, 
		num_finest_elems
	) );

	CHECK_CUDA_ERROR( cub::DeviceSelect::Flagged
	(
		d_temp_storage, 
		temp_storage_bytes, 
		d_buf_neighbours.west.act_idcs,
		d_compaction_flags.south_west, 
		d_neighbours.west.act_idcs,
		d_assem_sol_len, 
		num_finest_elems
	) );

	if (solver_params.solver_type == MWDG2)
	{
		CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged
		(
			d_temp_storage,
			temp_storage_bytes,
			d_buf_neighbours.north.levels,
			d_compaction_flags.north_east,
			d_neighbours.north.levels,
			d_assem_sol_len,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged
		(
			d_temp_storage,
			temp_storage_bytes,
			d_buf_neighbours.east.levels,
			d_compaction_flags.north_east,
			d_neighbours.east.levels,
			d_assem_sol_len,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged
		(
			d_temp_storage,
			temp_storage_bytes,
			d_buf_neighbours.south.levels,
			d_compaction_flags.south_west,
			d_neighbours.south.levels,
			d_assem_sol_len,
			num_finest_elems
		));

		CHECK_CUDA_ERROR(cub::DeviceSelect::Flagged
		(
			d_temp_storage,
			temp_storage_bytes,
			d_buf_neighbours.west.levels,
			d_compaction_flags.south_west,
			d_neighbours.west.levels,
			d_assem_sol_len,
			num_finest_elems
		));
	}

	free_device(d_temp_storage);
	free_device(d_assem_sol_len);
	delete      h_sol_len;
}