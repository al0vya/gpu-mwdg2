#include "write_soln_planar_fv1.cuh"

void write_soln_planar_fv1
(
    const char*              respath,
	const AssembledSolution& d_assem_sol,
	      real*              d_dt_CFL,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
)
{
    char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%s%d%s", respath, "planar-", saveint.count - 1, ".csv");

	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error writing planar data file, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	printf("Writing planar data.\n");

	fprintf
	(
		fp,
		"lower_left_x,"
		"lower_left_y,"
		"upper_right_x,"
		"upper_right_y,"
		"h0,"
		"qx0,"
		"qy0,"
		"z0"
		"\n"
	);
	
	real*           h0       = new real[d_assem_sol.length];
	real*           qx0      = new real[d_assem_sol.length];
	real*           qy0      = new real[d_assem_sol.length];
	real*           z0       = new real[d_assem_sol.length];
	real*           dt       = new real[d_assem_sol.length];
	HierarchyIndex* act_idcs = new HierarchyIndex[d_assem_sol.length];
	int*            levels   = new int[d_assem_sol.length];

	size_t bytes_flow     = d_assem_sol.length * sizeof(real);
	size_t bytes_act_idcs = d_assem_sol.length * sizeof(HierarchyIndex);
	size_t bytes_levels   = d_assem_sol.length * sizeof(int);
	
	copy(h0,       d_assem_sol.h0,       bytes_flow);
	copy(qx0,      d_assem_sol.qx0,      bytes_flow);
	copy(qy0,      d_assem_sol.qy0,      bytes_flow);
	copy(z0,       d_assem_sol.z0,       bytes_flow);
	copy(dt,       d_dt_CFL,             bytes_flow);
	copy(act_idcs, d_assem_sol.act_idcs, bytes_act_idcs);
	copy(levels,   d_assem_sol.levels,   bytes_levels);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		HierarchyIndex act_idx  = act_idcs[i];
		int            level    = levels[i];

		MortonCode code = act_idx - get_lvl_idx(level);

		real local_cell_size_x = dx_finest * ( 1 << (solver_params.L - level) );
		real local_cell_size_y = dy_finest * ( 1 << (solver_params.L - level) );

		Coordinate x = compact(code);
		Coordinate y = compact(code >> 1);

		real x_centre = x * local_cell_size_x + local_cell_size_x / C(2.0);
		real y_centre = y * local_cell_size_y + local_cell_size_y / C(2.0);

		Points points =
		{
			sim_params.xmin + x * local_cell_size_x,       // lower left  x
			sim_params.ymin + y * local_cell_size_y,       // lower left  y
			C(0.0),                                        // upper left  x
			C(0.0),                                        // upper left  y
			C(0.0),                                        // lower right x
			C(0.0),                                        // lower right y
			sim_params.xmin + (x + 1) * local_cell_size_x, // upper right x
			sim_params.ymin + (y + 1) * local_cell_size_y  // upper right y
		};

		fprintf
		(
		    fp,
			"%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT ","
			"%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT ","
		    "%" NUM_FIG NUM_FRMT
			"\n",
			points.ll_x,
			points.ll_y,
			points.ur_x,
			points.ur_y,
			h0 [i],
			qx0[i],
			qy0[i],
			z0 [i]
		);
	}

	delete[] h0;
	delete[] qx0;
	delete[] qy0;
	delete[] z0;
	delete[] dt;
	delete[] act_idcs;
	delete[] levels;
}