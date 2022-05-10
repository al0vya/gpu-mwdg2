#include "write_gauge_point_data.cuh"

__host__
void write_gauge_point_data
(
	const char*           respath,
	const int&            mesh_dim,
	const SolverParams&   solver_params,
	const PlottingParams& plot_params,
	AssembledSolution     d_plot_assem_sol,
	FinestGrid            p_finest_grid,
	GaugePoints           gauge_points,
	const real&           time_now,
	const real&           dx_finest,
	const real&           dy_finest,
	const bool&           first_t_step
)
{
	if (gauge_points.num_points == 0) return;
	
	int num_finest_elems = mesh_dim * mesh_dim;
	
	size_t bytes = sizeof(real) * num_finest_elems;

	copy(p_finest_grid.h, d_plot_assem_sol.h0, bytes);
	copy(p_finest_grid.z, d_plot_assem_sol.z0, bytes);
	
	if (plot_params.voutput_stage)
	{
		copy(p_finest_grid.qx, d_plot_assem_sol.qx0, bytes);
		copy(p_finest_grid.qy, d_plot_assem_sol.qy0, bytes);
	}

	char fullpath_h[255] = {"\0"};
	char fullpath_v[255] = {"\0"};

	sprintf(fullpath_h, "%s%s", respath, "stage.wd");
	sprintf(fullpath_v, "%s%s", respath, "stage.vl");
	
	FILE* fp_h = fopen(fullpath_h, (first_t_step) ? "w" : "a");
	FILE* fp_v = fopen(fullpath_v, (first_t_step) ? "w" : "a");

	if (nullptr == fp_h || nullptr == fp_v)
	{
		fprintf(stderr, "Error opening stage results file, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	if (first_t_step)
	{
		fprintf(fp_h, "Stage output, depth (m).\n\n");
		fprintf(fp_h, "Stage information (stage,x,y,elev):\n");

		for (int point = 0; point < gauge_points.num_points; point++)
		{
			Coordinate i = get_i_index( gauge_points.codes[point] );
			Coordinate j = get_j_index( gauge_points.codes[point] );
			
			int idx = mesh_dim * j + i;

			fprintf
			(
				fp_h,
				"%d"           // stage point number
				" %" NUM_FRMT  // x coordinate
				" %" NUM_FRMT  // y coordinate
				" %" NUM_FRMT  // elevation (z) 
				"\n",
				point + 1,
				i * dx_finest + dx_finest / C(2.0),
				j * dy_finest + dy_finest / C(2.0),
				p_finest_grid.z[idx]
			);
		}
		
		fprintf(fp_h, "\nOutput, depths:\n");
		fprintf(fp_h, "Time; stages 1 to %d\n", gauge_points.num_points);
		
		if (plot_params.voutput_stage)
		{
		    fprintf(fp_v, "Stage output, velocity (m/s).\n\n");
		    fprintf(fp_v, "Stage information (stage,x,y,elev):\n");
		    
		    for (int point = 0; point < gauge_points.num_points; point++)
		    {
		    	Coordinate i = get_i_index( gauge_points.codes[point] );
		    	Coordinate j = get_j_index( gauge_points.codes[point] );
		    	
		    	int idx = mesh_dim * j + i;
		    
		    	fprintf
		    	(
		    		fp_v,
		    		"%d"           // stage point number
		    		" %" NUM_FRMT  // x coordinate
		    		" %" NUM_FRMT  // y coordinate
		    		" %" NUM_FRMT  // elevation (z) 
		    		"\n",
		    		point + 1,
		    		i * dx_finest + dx_finest / C(2.0),
		    		j * dy_finest + dy_finest / C(2.0),
		    		p_finest_grid.z[idx]
		    	);
		    }
		    
		    fprintf(fp_v, "\nOutput, velocities:\n");
		    fprintf(fp_v, "Time; stages 1 to %d\n", gauge_points.num_points);
		}
	}

	fprintf(fp_h, "%" NUM_FRMT " ", time_now);

	if (plot_params.voutput_stage)
	{
		fprintf(fp_v, "%" NUM_FRMT " ", time_now);
	}

	for (int point = 0; point < gauge_points.num_points; point++)
	{
		Coordinate i = get_i_index( gauge_points.codes[point] );
		Coordinate j = get_j_index( gauge_points.codes[point] );

		int idx = j * mesh_dim + i;

		fprintf
		(
			fp_h,
			( (point + 1) == gauge_points.num_points )
			? "%" NUM_FRMT
			: "%" NUM_FRMT " ",
			p_finest_grid.h[idx]
		);

		if (plot_params.voutput_stage)
		{
			real vx = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qx[idx] / p_finest_grid.h[idx];
			real vy = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qy[idx] / p_finest_grid.h[idx];

			real speed = sqrt(vx * vx + vy * vy);

			fprintf
			(
				fp_v,
				( (point + 1) == gauge_points.num_points )
				? "%" NUM_FRMT
				: "%" NUM_FRMT " ",
				speed
			);
		}
	}

	fprintf(fp_h, "\n");
	
	if (plot_params.voutput_stage)
	{
		fprintf(fp_v, "\n");
	}

	fclose(fp_h);
	fclose(fp_v);
}