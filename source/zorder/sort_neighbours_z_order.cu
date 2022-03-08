#include "sort_neighbours_z_order.cuh"

__global__
void sort_neighbours_z_order
(
	const Neighbours   d_neighbours,
	const Neighbours   d_buf_neighbours,
	MortonCode*        d_rev_z_order,
	const int          num_finest_elems,
	const SolverParams solver_params
)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= num_finest_elems) return;

	const int sorted_idx = d_rev_z_order[idx];

	d_buf_neighbours.north.act_idcs[idx] = d_neighbours.north.act_idcs[sorted_idx];
	d_buf_neighbours.east.act_idcs[idx]  = d_neighbours.east.act_idcs[sorted_idx];
	d_buf_neighbours.south.act_idcs[idx] = d_neighbours.south.act_idcs[sorted_idx];
	d_buf_neighbours.west.act_idcs[idx]  = d_neighbours.west.act_idcs[sorted_idx];

	if (solver_params.solver_type == MWDG2)
	{
		d_buf_neighbours.north.levels[idx] = d_neighbours.north.levels[sorted_idx];
		d_buf_neighbours.east.levels[idx]  = d_neighbours.east.levels[sorted_idx];
		d_buf_neighbours.south.levels[idx] = d_neighbours.south.levels[sorted_idx];
		d_buf_neighbours.west.levels[idx]  = d_neighbours.west.levels[sorted_idx];
	}
}