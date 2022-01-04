#include "write_gauge_point_data.cuh"

void write_gauge_point_data
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParams&  solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol,
	GaugePoints              gauge_points,
	const real&              time_now,
	const bool&              first_t_step
)
{
	if (gauge_points.num_points == 0) return;
	
	int num_finest_elems = mesh_dim * mesh_dim;
	
	size_t bytes = sizeof(real) * num_finest_elems;

	real* h  = new real[num_finest_elems];
	real* qx = new real[num_finest_elems];
	real* qy = new real[num_finest_elems];
	real* z  = new real[num_finest_elems];

	copy(h,  d_plot_assem_sol.h0,  bytes);
	copy(qx, d_plot_assem_sol.qx0, bytes);
	copy(qy, d_plot_assem_sol.qy0, bytes);
	copy(z,  d_plot_assem_sol.z0,  bytes);
	
	char fullpath[255];

	sprintf(fullpath, "%s%s", respath, "stage.wd");
	
	FILE* fp = (first_t_step) ? fopen(fullpath, "w") : fopen(fullpath, "a");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening stage results file, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	if (first_t_step)
	{
		fprintf(fp, "time,");
		
		for (int i = 0; i < gauge_points.num_points; i++)
		{
			fprintf(fp, ( (i + 1) == gauge_points.num_points) ? "stage%d" : "stage%d,", i + 1);
		}

		fprintf(fp, "\n");
	}

	fprintf(fp, "%" NUM_FRMT ",", time_now);

	for (int i = 0; i < gauge_points.num_points; i++)
	{
		Coordinate x = compact(gauge_points.codes[i]);
		Coordinate y = compact(gauge_points.codes[i] >> 1);

		HierarchyIndex idx = y * mesh_dim + x;

		fprintf( fp, ( (i + 1) == gauge_points.num_points ) ? "%" NUM_FRMT : "%" NUM_FRMT ",", h[idx] + z[idx] );
	}

	fprintf(fp, "\n");

	fclose(fp);

	delete[] h;
	delete[] qx;
	delete[] qy;
	delete[] z;
}