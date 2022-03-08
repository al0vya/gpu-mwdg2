#include "rev_z_order_act_idcs.cuh"

__global__
void rev_z_order_act_idcs
(
	MortonCode*       d_rev_row_major,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	const int         num_finest_elems
)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= num_finest_elems) return;

	const int sorted_idx = d_rev_row_major[idx];

	d_assem_sol.act_idcs[idx] = d_buf_assem_sol.act_idcs[sorted_idx];
	d_assem_sol.levels[idx]   = d_buf_assem_sol.levels[sorted_idx];
}