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

	MortonCode code = point_sources.d_codes[idx];

	int src_type = point_sources.d_src_types[idx];

	if (src_type == HFIX || src_type == HVAR)
	{
		// indexing into the non-uniform grid with the Morton code is okay from some reason...?
		d_assem_sol.h0[code] = point_sources.d_srcs[idx] - d_assem_sol.z0[idx];
	}
	else if (src_type == QFIX || src_type == QVAR)
	{
		d_assem_sol.h0[code] += point_sources.q_src(dt, dx_finest, idx);
	}
}