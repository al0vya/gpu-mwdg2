#include "encode_topography_kernel_mw.cuh"

__global__
void encode_topography_kernel_mw
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

	real* z0  = &d_scale_coeffs.z0 [child_idx + 0];
	real* z1x = &d_scale_coeffs.z1x[child_idx + 0];
	real* z1y = &d_scale_coeffs.z1y[child_idx + 0];
	
	ScaleChildrenMW z_children =
	{
		{  z0[0],  z0[1],  z0[2],  z0[3] },
		{ z1x[0], z1x[1], z1x[2], z1x[3] },
		{ z1y[0], z1y[1], z1y[2], z1y[3] }
	};
	
	d_scale_coeffs.z0[parent_idx]  = encode_scale_0 (z_children);
	d_scale_coeffs.z1x[parent_idx] = encode_scale_1x(z_children);
	d_scale_coeffs.z1y[parent_idx] = encode_scale_1y(z_children);

	SubDetailMW z_details = encode_detail(z_children);

	d_details.z0.alpha[parent_idx]  = z_details._0.alpha;
	d_details.z0.beta[parent_idx]   = z_details._0.beta;
	d_details.z0.gamma[parent_idx]  = z_details._0.gamma;
	d_details.z1x.alpha[parent_idx] = z_details._1x.alpha;
	d_details.z1x.beta[parent_idx]  = z_details._1x.beta;
	d_details.z1x.gamma[parent_idx] = z_details._1x.gamma;
	d_details.z1y.alpha[parent_idx] = z_details._1y.alpha;
	d_details.z1y.beta[parent_idx]  = z_details._1y.beta;
	d_details.z1y.gamma[parent_idx] = z_details._1y.gamma;

	if ( (z_details.get_max() / maxes.z) >= epsilon_local ) d_preflagged_details[parent_idx] = SIGNIFICANT;
}