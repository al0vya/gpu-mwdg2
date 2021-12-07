#include "insert_point_srcs.cuh"

__global__
void insert_point_srcs
(
	AssembledSolution d_assem_sol,
	PointSources      point_sources,
	real              dt,
	real              dx_finest
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= point_sources.num_srcs) return;

	HierarchyIndex h_idx = point_sources.d_codes[idx];

	int src_type = point_sources.d_src_types[idx];

	if (src_type == HFIX || src_type == HVAR)
	{
		d_assem_sol.h0[h_idx] = point_sources.d_srcs[idx];
	}
	else if (src_type == QFIX || src_type == QVAR)
	{
		d_assem_sol.h0[h_idx] += point_sources.q_src(dt, dx_finest, idx);
	}
}