#include "write_gauge_point_data.cuh"

__host__
void write_gauge_point_data
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParams&      solver_params,
	const PlottingParams&    plot_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol,
	FinestGrid               p_finest_grid,
	GaugePoints              gauge_points,
	const real&              time_now,
	const bool&              first_t_step
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

	char fullpath_h[255];
	char fullpath_v[255];

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
		fprintf(fp_h, "time,");
		fprintf(fp_v, "time,");
		
		for (int i = 0; i < gauge_points.num_points; i++)
		{
			fprintf(fp_h, ( (i + 1) == gauge_points.num_points) ? "stage%d" : "stage%d,", i + 1);
			fprintf(fp_v, ( (i + 1) == gauge_points.num_points) ? "stage%d" : "stage%d,", i + 1);
		}

		fprintf(fp_h, "\n");
		fprintf(fp_v, "\n");
	}

	fprintf(fp_h, "%" NUM_FRMT ",", time_now);

	if (plot_params.voutput_stage)
	{
		fprintf(fp_v, "%" NUM_FRMT ",", time_now);
	}

	for (int i = 0; i < gauge_points.num_points; i++)
	{
		Coordinate x = compact( gauge_points.codes[i] );
		Coordinate y = compact( gauge_points.codes[i] >> 1 );

		int idx = y * mesh_dim + x;

		fprintf
		(
			fp_h,
			( (i + 1) == gauge_points.num_points )
			? "%" NUM_FRMT
			: "%" NUM_FRMT ",",
			p_finest_grid.h[idx] + p_finest_grid.z[idx]
		);

		if (plot_params.voutput_stage)
		{
			real vx = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qx[idx] / p_finest_grid.h[idx];
			real vy = (p_finest_grid.h[idx] < solver_params.tol_h) ? C(0.0) : p_finest_grid.qy[idx] / p_finest_grid.h[idx];

			real speed = sqrt(vx * vx + vy * vy);

			fprintf
			(
				fp_v,
				( (i + 1) == gauge_points.num_points )
				? "%" NUM_FRMT
				: "%" NUM_FRMT ",",
				speed
			);
		}
	}

	fprintf(fp_h, "\n");
	fprintf(fp_v, "\n");

	fclose(fp_h);
	fclose(fp_v);
}