#include "rev_z_order_act_idcs.cuh"

__global__
void rev_z_order_act_idcs
(
	MortonCode*       d_morton_codes,
	MortonCode*       d_indices,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	int               array_length
)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	const int sorted_idx = d_morton_codes[idx];

	d_assem_sol.act_idcs[idx] = d_buf_assem_sol.act_idcs[sorted_idx];
	d_assem_sol.levels[idx]   = d_buf_assem_sol.levels[sorted_idx];
}