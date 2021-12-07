#include "write_soln_vtk.cuh"

__host__ void write_soln_vtk
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

	sprintf(fullpath, "%s%s%d%s", respath, "results-", saveint.count - 1, ".vtk");

	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error writing VTK file, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	printf("Writing VTK file.\n");

	fprintf
	(
		fp,
		"# vtk DataFile Version 3.0\n"
		"Multiresolution flow and topo data\n"
		"ASCII\n"
		"\n"
		"DATASET UNSTRUCTURED_GRID\n"
		"POINTS %d float\n",
		d_assem_sol.length * 4
	);

	// xmax, ymax from sim_params may have non-zero origin
	// however, Morton codes assume zero origin
	// hence, modified xmax, ymax for bounds checking
	real xmax_0_orig = sim_params.xsz * dx_finest;
	real ymax_0_orig = sim_params.ysz * dx_finest;

	real*           h        = new real[d_assem_sol.length];
	real*           qx       = new real[d_assem_sol.length];
	real*           qy       = new real[d_assem_sol.length];
	real*           z        = new real[d_assem_sol.length];
	real*           dt       = new real[d_assem_sol.length];
	HierarchyIndex* act_idcs = new HierarchyIndex[d_assem_sol.length];
	int*            levels   = new int[d_assem_sol.length];

	size_t bytes_flow     = d_assem_sol.length * sizeof(real);
	size_t bytes_act_idcs = d_assem_sol.length * sizeof(HierarchyIndex);
	size_t bytes_levels   = d_assem_sol.length * sizeof(int);
	
	copy(h,        d_assem_sol.h0,       bytes_flow);
	copy(qx,       d_assem_sol.qx0,      bytes_flow);
	copy(qy,       d_assem_sol.qy0,      bytes_flow);
	copy(z,        d_assem_sol.z0,       bytes_flow);
	copy(dt,       d_dt_CFL,             bytes_flow);
	copy(act_idcs, d_assem_sol.act_idcs, bytes_act_idcs);
	copy(levels,   d_assem_sol.levels,   bytes_levels);

	// number of cells excluding those in extended domain
	int num_bound = 0;

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		HierarchyIndex act_idx  = act_idcs[i];
		int      level    = levels[i];

		MortonCode code = act_idx - get_lvl_idx(level);

		real local_cell_size_x = dx_finest * ( 1 << (solver_params.L - level) );
		real local_cell_size_y = dy_finest * ( 1 << (solver_params.L - level) );

		Coordinate x = compact(code);
		Coordinate y = compact(code >> 1);

		real x_centre = x * local_cell_size_x + local_cell_size_x / C(2.0);
		real y_centre = y * local_cell_size_y + local_cell_size_y / C(2.0);

		bool bound = (x_centre < xmax_0_orig && y_centre < ymax_0_orig);

		//if (!bound) continue;

		num_bound++;

		Points points =
		{
			sim_params.xmin + x * local_cell_size_x,       // lower left  x
			sim_params.ymin + y * local_cell_size_y,       // lower left  y
			sim_params.xmin + x * local_cell_size_x,       // upper left  x
			sim_params.ymin + (y + 1) * local_cell_size_y, // upper left  y
			sim_params.xmin + (x + 1) * local_cell_size_x, // lower right x
			sim_params.ymin + y * local_cell_size_y,       // lower right y
			sim_params.xmin + (x + 1) * local_cell_size_x, // upper right x
			sim_params.ymin + (y + 1) * local_cell_size_y  // upper right y
		};

		fprintf
		(
			fp,
			"%" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT "\n"
			"%" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT "\n"
			"%" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT "\n"
			"%" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT " %" NUM_FIG NUM_FRMT "\n",
			points.ll_x, points.ll_y, C(1.0),
			points.ul_x, points.ul_y, C(1.0),
			points.lr_x, points.lr_y, C(1.0),
			points.ur_x, points.ur_y, C(1.0)
		);
	}

	fprintf(fp, "\nCELLS %d %d\n", d_assem_sol.length, d_assem_sol.length * 5);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		// point counter to make sure correct vertices per cell
		int pt_ctr = i * 4;

		fprintf(fp, "4 %d %d %d %d\n", 0 + pt_ctr, 1 + pt_ctr, 2 + pt_ctr, 3 + pt_ctr);
	}

	fprintf(fp, "\nCELL_TYPES %d\n", d_assem_sol.length);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		fprintf(fp, "%d\n", 8);
	}

	fprintf(fp, "\nCELL_DATA %d\n", d_assem_sol.length);

	fprintf
	(
		fp,
		"\nSCALARS h float 1\n"
		"LOOKUP_TABLE default\n"
	);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		fprintf( fp, "%" NUM_FIG NUM_FRMT "\n", h[i] );
	}
	
	fprintf
	(
		fp,
		"\nSCALARS dt float 1\n"
		"LOOKUP_TABLE default\n"
	);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		fprintf( fp, "%" NUM_FIG NUM_FRMT "\n", dt[i] );
	}

	fprintf
	(
		fp,
		"\nSCALARS qx float 1\n"
		"LOOKUP_TABLE default\n"
	);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		real vx = (h[i] < solver_params.tol_h) ? C(0.0) : qx[i] / h[i];
		
		fprintf( fp, "%" NUM_FIG NUM_FRMT "\n", qx[i]);
	}

	fprintf
	(
		fp,
		"\nSCALARS qy float 1\n"
		"LOOKUP_TABLE default\n"
	);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		real vy = (h[i] < solver_params.tol_h) ? C(0.0) : qy[i] / h[i];
		
		fprintf( fp, "%" NUM_FIG NUM_FRMT "\n", qy[i]);
	}

	fprintf
	(
		fp,
		"\nSCALARS z float 1\n"
		"LOOKUP_TABLE default\n"
	);

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		fprintf( fp, "%" NUM_FIG NUM_FRMT "\n", z[i] );
	}

	fclose(fp);

	delete[] h;
	delete[] qx;
	delete[] qy;
	delete[] z;
	delete[] dt;
	delete[] act_idcs;
	delete[] levels;
}