#include "write_all_raster_maps_fv1.cuh"

void write_all_raster_maps_fv1
(
    const char*              respath,
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

	real* h  = new real[num_finest_cells];
	real* qx = new real[num_finest_cells];
	real* qy = new real[num_finest_cells];
	real* z  = new real[num_finest_cells];

	real*           h0       = new real[d_assem_sol.length];
	real*           qx0      = new real[d_assem_sol.length];
	real*           qy0      = new real[d_assem_sol.length];
	real*           z0       = new real[d_assem_sol.length];
	HierarchyIndex* act_idcs = new HierarchyIndex[d_assem_sol.length];
	int*            levels   = new int[d_assem_sol.length];

	size_t bytes_flow     = d_assem_sol.length * sizeof(real);
	size_t bytes_act_idcs = d_assem_sol.length * sizeof(HierarchyIndex);
	size_t bytes_levels   = d_assem_sol.length * sizeof(int);
	
	copy(h0,       d_assem_sol.h0,       bytes_flow);
	copy(qx0,      d_assem_sol.qx0,      bytes_flow);
	copy(qy0,      d_assem_sol.qy0,      bytes_flow);
	copy(z0,       d_assem_sol.z0,       bytes_flow);
	copy(act_idcs, d_assem_sol.act_idcs, bytes_act_idcs);
	copy(levels,   d_assem_sol.levels,   bytes_levels);

	for (int element = 0; element < d_assem_sol.length; element++)
	{
		int level = levels[element];

		MortonCode code = act_idcs[element] - get_lvl_idx(level);

		code <<= 2 * (solver_params.L - level);

		Coordinate i = get_i_index(code);
		Coordinate j = get_j_index(code);

		int side_len = 1 << (solver_params.L - level);

		for (int j_loc = 0; j_loc < side_len; j_loc++)
		{
			for (int i_loc = 0; i_loc < side_len; i_loc++)
			{
				int idx = (mesh_dim) * (j + j_loc) + (i + i_loc);

				h[idx]  = h0 [element];
				qx[idx] = qx0[element];
				qy[idx] = qy0[element];
				z[idx]  = z0 [element];
			}
		}
	}
	
	write_raster_file
	(
		respath,
		"wd",
		h,
		sim_params,
		saveint,
		dx_finest,
		mesh_dim
	);

	write_raster_file
	(
		respath,
		"qx",
		qx,
		sim_params,
		saveint,
		dx_finest,
		mesh_dim
	);

	write_raster_file
	(
		respath,
		"qy",
		qy,
		sim_params,
		saveint,
		dx_finest,
		mesh_dim
	);

	delete[] h;
	delete[] qx;
	delete[] qy;
	delete[] z;

	delete[] h0;
	delete[] qx0;
	delete[] qy0;
	delete[] z0;
	delete[] act_idcs;
	delete[] levels;
}