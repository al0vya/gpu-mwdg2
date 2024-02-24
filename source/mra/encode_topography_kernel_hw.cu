#include "encode_topography_kernel_hw.cuh"

__global__
void encode_topography_kernel_hw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams      solver_params,
	int               level
)
{
	MortonCode idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int num_threads = 1 << (2 * level);

	if (idx >= num_threads) return;

	HierarchyIndex prev_lvl_idx = get_lvl_idx(level - 1);
	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	HierarchyIndex parent_idx = curr_lvl_idx + idx;
	HierarchyIndex child_idx  = next_lvl_idx + 4 * idx;

	real epsilon_local = solver_params.epsilon / ( 1 << (solver_params.L - level) );

	ScaleChildrenHW z_children =
	{
		d_scale_coeffs.z0[child_idx + 0],
		d_scale_coeffs.z0[child_idx + 1],
		d_scale_coeffs.z0[child_idx + 2],
		d_scale_coeffs.z0[child_idx + 3]
	};

	SubDetailHW z_details =
	{
		encode_detail_alpha(z_children),
		encode_detail_beta (z_children),
		encode_detail_gamma(z_children)
	};

	d_details.z0.alpha[parent_idx] = z_details.alpha;
	d_details.z0.beta[parent_idx]  = z_details.beta;
	d_details.z0.gamma[parent_idx] = z_details.gamma;

	d_scale_coeffs.z0[parent_idx] = encode_scale(z_children);

	if ( (z_details.get_max() / maxes.z) >= epsilon_local ) d_preflagged_details[parent_idx] = SIGNIFICANT;
}