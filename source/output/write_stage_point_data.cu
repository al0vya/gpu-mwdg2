#include "write_stage_point_data.cuh"

__host__
void write_stage_point_data
(
	const int&            mesh_dim,
	const SolverParams&   solver_params,
	const PlottingParams& plot_params,
	AssembledSolution     d_plot_assem_sol,
	FinestGrid            p_finest_grid,
	StagePoints           stage_points,
	const real&           current_time,
	const real&           dx_finest,
	const real&           dy_finest,
	const bool&           first_t_step
)
{
	if (stage_points.num_points == 0) return;
	
	int num_finest_elems = mesh_dim * mesh_dim;
	
	size_t bytes = sizeof(real) * num_finest_elems;

	copy_cuda(p_finest_grid.h, d_plot_assem_sol.h0, bytes);
	copy_cuda(p_finest_grid.z, d_plot_assem_sol.z0, bytes);
	
	if (plot_params.voutput_stage)
	{
		copy_cuda(p_finest_grid.qx, d_plot_assem_sol.qx0, bytes);
		copy_cuda(p_finest_grid.qy, d_plot_assem_sol.qy0, bytes);
	}

	char fullpath_h[255]  = {"\0"};
	char fullpath_vx[255] = {"\0"};
	char fullpath_vy[255] = {"\0"};

	sprintf(fullpath_h,  "%s%c%s%s", plot_params.dirroot, '/', plot_params.resroot, ".stage");
	sprintf(fullpath_vx, "%s%c%s%s", plot_params.dirroot, '/', plot_params.resroot, ".xvelocity");
	sprintf(fullpath_vy, "%s%c%s%s", plot_params.dirroot, '/', plot_params.resroot, ".yvelocity");
	
	FILE* fp_h  = fopen(fullpath_h,  (first_t_step) ? "w" : "a");
	FILE* fp_vx = fopen(fullpath_vx, (first_t_step) ? "w" : "a");
	FILE* fp_vy = fopen(fullpath_vy, (first_t_step) ? "w" : "a");

	if (NULL == fp_h || NULL == fp_vx || NULL == fp_vy)
	{
		fprintf(stderr, "Error opening stage results file, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	if (first_t_step)
	{
		fprintf(fp_h, "Stage output, depth (m).\n\n");
		fprintf(fp_h, "Stage information (stage,x,y,elev):\n");

		for (int point = 0; point < stage_points.num_points; point++)
		{
			Coordinate i = get_i_index( stage_points.codes[point] );
			Coordinate j = get_j_index( stage_points.codes[point] );
			
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
		fprintf(fp_h, "Time; stages 1 to %d\n", stage_points.num_points);
		
		if (plot_params.voutput_stage)
		{
		    fprintf(fp_vx, "Stage output, x velocity (m/s).\n\n");
		    fprintf(fp_vx, "Stage information (stage,x,y,elev):\n");
		   
			fprintf(fp_vy, "Stage output, y velocity (m/s).\n\n");
		    fprintf(fp_vy, "Stage information (stage,x,y,elev):\n");
		    
		    for (int point = 0; point < stage_points.num_points; point++)
		    {
		    	Coordinate i = get_i_index( stage_points.codes[point] );
		    	Coordinate j = get_j_index( stage_points.codes[point] );
		    	
		    	int idx = mesh_dim * j + i;
		    
		    	fprintf
		    	(
		    		fp_vx,
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

				fprintf
		    	(
		    		fp_vy,
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
		    
		    fprintf(fp_vx, "\nOutput, x velocities:\n");
		    fprintf(fp_vx, "Time; stages 1 to %d\n", stage_points.num_points);
			
			fprintf(fp_vy, "\nOutput, y velocities:\n");
		    fprintf(fp_vy, "Time; stages 1 to %d\n", stage_points.num_points);
		}
	}

	fprintf(fp_h, "%" NUM_FRMT " ", current_time);

	if (plot_params.voutput_stage)
	{
		fprintf(fp_vx, "%" NUM_FRMT " ", current_time);
		fprintf(fp_vy, "%" NUM_FRMT " ", current_time);
	}

	for (int point = 0; point < stage_points.num_points; point++)
	{
		Coordinate i = get_i_index( stage_points.codes[point] );
		Coordinate j = get_j_index( stage_points.codes[point] );

		int idx = j * mesh_dim + i;

		fprintf
		(
			fp_h,
			( (point + 1) == stage_points.num_points )
			? "%" NUM_FRMT
			: "%" NUM_FRMT " ",
			p_finest_grid.h[idx]
		);

		if (plot_params.voutput_stage)
		{
			real vx = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qx[idx] / p_finest_grid.h[idx];
			real vy = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qy[idx] / p_finest_grid.h[idx];

			fprintf
			(
				fp_vx,
				( (point + 1) == stage_points.num_points )
				? "%" NUM_FRMT
				: "%" NUM_FRMT " ",
				vx
			);

			fprintf
			(
				fp_vy,
				( (point + 1) == stage_points.num_points )
				? "%" NUM_FRMT
				: "%" NUM_FRMT " ",
				vy
			);
		}
	}

	fprintf(fp_h, "\n");
	
	if (plot_params.voutput_stage)
	{
		fprintf(fp_vx, "\n");
		fprintf(fp_vy, "\n");
	}

	fclose(fp_h);
	fclose(fp_vx);
	fclose(fp_vy);
}