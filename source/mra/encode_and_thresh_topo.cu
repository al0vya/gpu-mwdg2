#include "encode_and_thresh_topo.cuh"

__global__
void encode_and_thresh_topo
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams  solver_params,
	SimulationParams sim_params,
	int               level,
	bool              first_time_step
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

	if (solver_params.solver_type == HWFV1)
	{
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
	else if (solver_params.solver_type == MWDG2)
	{
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

		SubDetailMW z_details_mw = encode_detail(z_children);

		d_details.z0.alpha[parent_idx] =  z_details_mw._0.alpha;
		d_details.z0.beta[parent_idx]  =  z_details_mw._0.beta;
		d_details.z0.gamma[parent_idx] =  z_details_mw._0.gamma;
		d_details.z1x.alpha[parent_idx] = z_details_mw._1x.alpha;
		d_details.z1x.beta[parent_idx]  = z_details_mw._1x.beta;
		d_details.z1x.gamma[parent_idx] = z_details_mw._1x.gamma;
		d_details.z1y.alpha[parent_idx] = z_details_mw._1y.alpha;
		d_details.z1y.beta[parent_idx]  = z_details_mw._1y.beta;
		d_details.z1y.gamma[parent_idx] = z_details_mw._1y.gamma;

		if ((z_details_mw.get_max() / maxes.z) >= epsilon_local) {
			d_preflagged_details[parent_idx] = SIGNIFICANT;

			if (solver_params.grading) {
				MortonCode code = idx;

				Coordinate i = compact(code);
				Coordinate j = compact(code >> 1);

				if ((i > 0 && i < sim_params.xsz) && (j > 0 && j < sim_params.ysz)) {

					Coordinate i_n = i;
					Coordinate i_e = i + 1;
					Coordinate i_s = i;
					Coordinate i_w = i - 1;

					Coordinate j_n = j + 1;
					Coordinate j_e = j;
					Coordinate j_s = j - 1;
					Coordinate j_w = j;

					HierarchyIndex n_idx = generate_morton_code(i_n, j_n);
					HierarchyIndex e_idx = generate_morton_code(i_e, j_e);
					HierarchyIndex s_idx = generate_morton_code(i_s, j_s);
					HierarchyIndex w_idx = generate_morton_code(i_w, j_w);

					d_preflagged_details[curr_lvl_idx + n_idx] = SIGNIFICANT;
					d_preflagged_details[curr_lvl_idx + e_idx] = SIGNIFICANT;
					d_preflagged_details[curr_lvl_idx + s_idx] = SIGNIFICANT;
					d_preflagged_details[curr_lvl_idx + w_idx] = SIGNIFICANT;
				}
			}
		}
	}
}