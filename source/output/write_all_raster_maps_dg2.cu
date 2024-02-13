#include "write_all_raster_maps_dg2.cuh"

void write_all_raster_maps_dg2
(
    const PlottingParams&    plot_params,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint,
	const bool&              first_t_step
)
{
    const int mesh_dim = 1 << (solver_params.L);

	const int num_finest_cells = mesh_dim * mesh_dim;

	real* h    = new real[num_finest_cells];
	real* qx   = new real[num_finest_cells];
	real* qy   = new real[num_finest_cells];
	real* vx   = new real[num_finest_cells];
	real* vy   = new real[num_finest_cells];
	real* z    = new real[num_finest_cells];
	real* eta  = new real[num_finest_cells];

	real*           h0       = new real[d_assem_sol.length];
	real*           h1x      = new real[d_assem_sol.length];
	real*           h1y      = new real[d_assem_sol.length];
	real*           qx0      = new real[d_assem_sol.length];
	real*           qx1x     = new real[d_assem_sol.length];
	real*           qx1y     = new real[d_assem_sol.length];
	real*           qy0      = new real[d_assem_sol.length];
	real*           qy1x     = new real[d_assem_sol.length];
	real*           qy1y     = new real[d_assem_sol.length];
	real*           z0       = new real[d_assem_sol.length];
	real*           z1x      = new real[d_assem_sol.length];
	real*           z1y      = new real[d_assem_sol.length];
	HierarchyIndex* act_idcs = new HierarchyIndex[d_assem_sol.length];
	int*            levels   = new int[d_assem_sol.length];

	size_t bytes_flow     = d_assem_sol.length * sizeof(real);
	size_t bytes_act_idcs = d_assem_sol.length * sizeof(HierarchyIndex);
	size_t bytes_levels   = d_assem_sol.length * sizeof(int);
	
	copy_cuda(h0,       d_assem_sol.h0,       bytes_flow);
	copy_cuda(h1x,      d_assem_sol.h1x,      bytes_flow);
	copy_cuda(h1y,      d_assem_sol.h1y,      bytes_flow);
	copy_cuda(qx0,      d_assem_sol.qx0,      bytes_flow);
	copy_cuda(qx1x,     d_assem_sol.qx1x,     bytes_flow);
	copy_cuda(qx1y,     d_assem_sol.qx1y,     bytes_flow);
	copy_cuda(qy0,      d_assem_sol.qy0,      bytes_flow);
	copy_cuda(qy1x,     d_assem_sol.qy1x,     bytes_flow);
	copy_cuda(qy1y,     d_assem_sol.qy1y,     bytes_flow);
	copy_cuda(z0,       d_assem_sol.z0,       bytes_flow);
	copy_cuda(z1x,      d_assem_sol.z1x,      bytes_flow);
	copy_cuda(z1y,      d_assem_sol.z1y,      bytes_flow);
	copy_cuda(act_idcs, d_assem_sol.act_idcs, bytes_act_idcs);
	copy_cuda(levels,   d_assem_sol.levels,   bytes_levels);

	for (int element = 0; element < d_assem_sol.length; element++)
	{
		int level = levels[element];

		MortonCode code = act_idcs[element] - get_lvl_idx(level);

		code <<= 2 * (solver_params.L - level);

		Coordinate i = get_i_index(code);
		Coordinate j = get_j_index(code);

		int side_len = 1 << (solver_params.L - level);

		real dx_unit = C(1.0) / side_len;

		FlowCoeffs coeffs =
		{
			{
				h0[element],
				h1x[element],
				h1y[element]
			},
			{
				qx0[element],
				qx1x[element],
				qx1y[element]
			},
			{
				qy0[element],
				qy1x[element],
				qy1y[element]
			}
		};

		PlanarCoefficients z_planar =
		{
			z0[element],
			z1x[element],
			z1y[element]
		};

		for (int j_loc = 0; j_loc < side_len; j_loc++)
		{
			real y_unit = j_loc * dx_unit + dx_unit / C(2.0);

			for (int i_loc = 0; i_loc < side_len; i_loc++)
			{
				real x_unit = i_loc * dx_unit + dx_unit / C(2.0);
				
				LegendreBasis leg_basis =
				{
					C(1.0),
					sqrt( C(3.0) ) * ( C(2.0) * x_unit - C(1.0) ),
					sqrt( C(3.0) ) * ( C(2.0) * y_unit - C(1.0) )
				};

				real h_loc_face_val = eval_loc_face_val_dg2(coeffs.h, leg_basis);
				real z_loc_face_val = eval_loc_face_val_dg2(z_planar, leg_basis);

				int idx = (mesh_dim) * (j + j_loc) + (i + i_loc);

				h[idx]   = h_loc_face_val;
				z[idx]   = z_loc_face_val;
				eta[idx] = h_loc_face_val + z_loc_face_val;

				// only compute velocity or discharge if voutput/qoutput params are included
				if (plot_params.voutput || plot_params.qoutput)
				{
					real qx_loc_face_val = eval_loc_face_val_dg2(coeffs.qx, leg_basis);
					real qy_loc_face_val = eval_loc_face_val_dg2(coeffs.qy, leg_basis);
					
					if (plot_params.qoutput)
					{
						qx[idx] = qx_loc_face_val;
						qy[idx] = qy_loc_face_val;
					}
					
					if (plot_params.voutput)
					{
						vx[idx] = (h_loc_face_val > solver_params.tol_h) ? qx_loc_face_val / h_loc_face_val : C(0.0);
						vy[idx] = (h_loc_face_val > solver_params.tol_h) ? qy_loc_face_val / h_loc_face_val : C(0.0);
					}
				}
			}
		}
	}

	if (!plot_params.elevoff)
	{
		write_raster_file
		(
			plot_params,
			"elev",
			eta,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);
	}

	if (!plot_params.depthoff)
	{
		write_raster_file
		(
			plot_params,
			"wd",
			h,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);
	}

	if (plot_params.qoutput)
	{
		write_raster_file
		(
			plot_params,
			"qx",
			qx,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			plot_params,
			"qy",
			qy,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);
	}

	if (plot_params.voutput)
	{
		write_raster_file
		(
			plot_params,
			"vx",
			vx,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			plot_params,
			"vy",
			vy,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);
	}

	if (first_t_step)
	{
		write_raster_file
		(
			plot_params,
			"dem",
			z,
			sim_params,
			saveint,
			dx_finest,
			mesh_dim
		);
	}

	delete[] h;
	delete[] qx;
	delete[] qy;
	delete[] vx;
	delete[] vy;
	delete[] z;
	delete[] eta;

	delete[] h0;
	delete[] h1x;
	delete[] h1y;
	delete[] qx0;
	delete[] qx1x;
	delete[] qx1y;
	delete[] qy0;
	delete[] qy1x;
	delete[] qy1y;
	delete[] z0;
	delete[] z1x;
	delete[] z1y;
	delete[] act_idcs;
	delete[] levels;
}