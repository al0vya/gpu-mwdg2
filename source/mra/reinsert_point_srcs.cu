#include "reinsert_point_srcs.cuh"

__global__
void reinsert_point_srcs
(
	ScaleCoefficients d_scale_coeffs,
	PointSources      point_sources,
	real              dt,
	real              dx_finest,
	int               max_ref_lvl
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= point_sources.num_srcs) return;

	HierarchyIndex h_idx = get_lvl_idx(max_ref_lvl) + point_sources.d_codes[idx];

	int src_type = point_sources.d_src_types[idx];

	if (src_type == HFIX || src_type == HVAR)
	{
		d_scale_coeffs.eta0[h_idx] = point_sources.d_srcs[idx];
	}
	else if (src_type == QFIX || src_type == QVAR)
	{
		d_scale_coeffs.eta0[h_idx] += point_sources.q_src(dt, dx_finest, idx);
	}
}