#include "friction_implicit.cuh"

__global__
void friction_implicit
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParams     solver_params,
	SimulationParams sim_params,
	real                 dt
)
{
	typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

	__shared__ union
	{
		typename block_scan::TempStorage temp_storage;
		HierarchyIndex indices[THREADS_PER_BLOCK];

	} shared;
	
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	int above_thresh = 0;

	int thread_prefix_sum = 0;

	int num_fric = 0;

	if (idx < d_assem_sol.length)
	{
		real  h  = d_assem_sol.h0[idx];
		real& qx = d_assem_sol.qx0[idx];
		real& qy = d_assem_sol.qy0[idx];

		int below_depth = (h < solver_params.tol_h);
		
		real ux = (below_depth) ? C(0.0) : qx / h;
		real uy = (below_depth) ? C(0.0) : qy / h;

		int below_speed = (abs(ux) < solver_params.tol_s && abs(uy) < solver_params.tol_s);

		if (below_depth || below_speed)
		{
			qx = C(0.0);
			qy = C(0.0);
		}

		above_thresh = !(below_depth || below_speed);
	}

	block_scan(shared.temp_storage).ExclusiveSum
	(
		above_thresh,
		thread_prefix_sum,
		num_fric
	);

	__syncthreads();

	if (above_thresh) shared.indices[thread_prefix_sum] = idx;

	__syncthreads();

	if (t_idx >= num_fric) return;

	idx = shared.indices[t_idx];

	real n = sim_params.manning;

	real  h   = d_assem_sol.h0[idx];
	real  h_n = d_neighbours.north.h0[idx];
	real  h_e = d_neighbours.east.h0[idx];
	real  h_s = d_neighbours.south.h0[idx];
	real  h_w = d_neighbours.west.h0[idx];

	real& qx   = d_assem_sol.qx0[idx];
	real& qx_n = d_neighbours.north.qx0[idx];
	real& qx_e = d_neighbours.east.qx0[idx];
	real& qx_s = d_neighbours.south.qx0[idx];
	real& qx_w = d_neighbours.west.qx0[idx];

	real& qy   = d_assem_sol.qy0[idx];
	real& qy_n = d_neighbours.north.qy0[idx];
	real& qy_e = d_neighbours.east.qy0[idx];
	real& qy_s = d_neighbours.south.qy0[idx];
	real& qy_w = d_neighbours.west.qy0[idx];
	
	const real tol_h = solver_params.tol_h;
	const real tol_s = solver_params.tol_s;

	if (solver_params.solver_type == HWFV1)
	{
		apply_friction(h,   tol_h, tol_s, qx,   qy,   sim_params.manning, sim_params.g, dt);
		apply_friction(h_n, tol_h, tol_s, qx_n, qy_n, sim_params.manning, sim_params.g, dt);
		apply_friction(h_e, tol_h, tol_s, qx_e, qy_e, sim_params.manning, sim_params.g, dt);
		apply_friction(h_s, tol_h, tol_s, qx_s, qy_s, sim_params.manning, sim_params.g, dt);
		apply_friction(h_w, tol_h, tol_s, qx_w, qy_w, sim_params.manning, sim_params.g, dt);
	}
	else if (solver_params.solver_type == MWDG2)
	{
		real  h1x   = d_assem_sol.h1x[idx];
		real  h1x_n = d_neighbours.north.h1x[idx];
		real  h1x_e = d_neighbours.east.h1x[idx];
		real  h1x_s = d_neighbours.south.h1x[idx];
		real  h1x_w = d_neighbours.west.h1x[idx];

		real& qx1x   = d_assem_sol.qx1x[idx];
		real& qx1x_n = d_neighbours.north.qx1x[idx];
		real& qx1x_e = d_neighbours.east.qx1x[idx];
		real& qx1x_s = d_neighbours.south.qx1x[idx];
		real& qx1x_w = d_neighbours.west.qx1x[idx];

		real& qy1x   = d_assem_sol.qy1x[idx];
		real& qy1x_n = d_neighbours.north.qy1x[idx];
		real& qy1x_e = d_neighbours.east.qy1x[idx];
		real& qy1x_s = d_neighbours.south.qy1x[idx];
		real& qy1x_w = d_neighbours.west.qy1x[idx];

		real  h1y   = d_assem_sol.h1y[idx];
		real  h1y_n = d_neighbours.north.h1y[idx];
		real  h1y_e = d_neighbours.east.h1y[idx];
		real  h1y_s = d_neighbours.south.h1y[idx];
		real  h1y_w = d_neighbours.west.h1y[idx];

		real& qx1y   = d_assem_sol.qx1y[idx];
		real& qx1y_n = d_neighbours.north.qx1y[idx];
		real& qx1y_e = d_neighbours.east.qx1y[idx];
		real& qx1y_s = d_neighbours.south.qx1y[idx];
		real& qx1y_w = d_neighbours.west.qx1y[idx];

		real& qy1y   = d_assem_sol.qy1y[idx];
		real& qy1y_n = d_neighbours.north.qy1y[idx];
		real& qy1y_e = d_neighbours.east.qy1y[idx];
		real& qy1y_s = d_neighbours.south.qy1y[idx];
		real& qy1y_w = d_neighbours.west.qy1y[idx];

		real h_gauss_lower_x   = h   - h1x;
		real h_gauss_lower_x_n = h_n - h1x_n;
		real h_gauss_lower_x_e = h_e - h1x_e;
		real h_gauss_lower_x_s = h_s - h1x_s;
		real h_gauss_lower_x_w = h_w - h1x_w;
		
		real h_gauss_upper_x   = h   + h1x;
		real h_gauss_upper_x_n = h_n + h1x_n;
		real h_gauss_upper_x_e = h_e + h1x_e;
		real h_gauss_upper_x_s = h_s + h1x_s;
		real h_gauss_upper_x_w = h_w + h1x_w;
		
		real h_gauss_lower_y   = h   - h1y;
		real h_gauss_lower_y_n = h_n - h1y_n;
		real h_gauss_lower_y_e = h_e - h1y_e;
		real h_gauss_lower_y_s = h_s - h1y_s;
		real h_gauss_lower_y_w = h_w - h1y_w;
		
		real h_gauss_upper_y   = h   + h1y;
		real h_gauss_upper_y_n = h_n + h1y_n;
		real h_gauss_upper_y_e = h_e + h1y_e;
		real h_gauss_upper_y_s = h_s + h1y_s;
		real h_gauss_upper_y_w = h_w + h1y_w;
		
		real qx_gauss_lower_x   = qx   - qx1x;
		real qx_gauss_lower_x_n = qx_n - qx1x_n;
		real qx_gauss_lower_x_e = qx_e - qx1x_e;
		real qx_gauss_lower_x_s = qx_s - qx1x_s;
		real qx_gauss_lower_x_w = qx_w - qx1x_w;
		
		real qx_gauss_upper_x   = qx   + qx1x;
		real qx_gauss_upper_x_n = qx_n + qx1x_n;
		real qx_gauss_upper_x_e = qx_e + qx1x_e;
		real qx_gauss_upper_x_s = qx_s + qx1x_s;
		real qx_gauss_upper_x_w = qx_w + qx1x_w;
		
		real qx_gauss_lower_y   = qx   - qx1y;
		real qx_gauss_lower_y_n = qx_n - qx1y_n;
		real qx_gauss_lower_y_e = qx_e - qx1y_e;
		real qx_gauss_lower_y_s = qx_s - qx1y_s;
		real qx_gauss_lower_y_w = qx_w - qx1y_w;
		
		real qx_gauss_upper_y   = qx   + qx1y;
		real qx_gauss_upper_y_n = qx_n + qx1y_n;
		real qx_gauss_upper_y_e = qx_e + qx1y_e;
		real qx_gauss_upper_y_s = qx_s + qx1y_s;
		real qx_gauss_upper_y_w = qx_w + qx1y_w;
		
		real qy_gauss_lower_x   = qy   - qy1x;
		real qy_gauss_lower_x_n = qy_n - qy1x_n;
		real qy_gauss_lower_x_e = qy_e - qy1x_e;
		real qy_gauss_lower_x_s = qy_s - qy1x_s;
		real qy_gauss_lower_x_w = qy_w - qy1x_w;
		
		real qy_gauss_upper_x   = qy   + qy1x;
		real qy_gauss_upper_x_n = qy_n + qy1x_n;
		real qy_gauss_upper_x_e = qy_e + qy1x_e;
		real qy_gauss_upper_x_s = qy_s + qy1x_s;
		real qy_gauss_upper_x_w = qy_w + qy1x_w;
		
		real qy_gauss_lower_y   = qy   - qy1y;
		real qy_gauss_lower_y_n = qy_n - qy1y_n;
		real qy_gauss_lower_y_e = qy_e - qy1y_e;
		real qy_gauss_lower_y_s = qy_s - qy1y_s;
		real qy_gauss_lower_y_w = qy_w - qy1y_w;
		
		real qy_gauss_upper_y   = qy   + qy1y;
		real qy_gauss_upper_y_n = qy_n + qy1y_n;
		real qy_gauss_upper_y_e = qy_e + qy1y_e;
		real qy_gauss_upper_y_s = qy_s + qy1y_s;
		real qy_gauss_upper_y_w = qy_w + qy1y_w;

		apply_friction(h,   tol_h, tol_s, qx,   qy,   sim_params.manning, sim_params.g, dt);
		apply_friction(h_n, tol_h, tol_s, qx_n, qy_n, sim_params.manning, sim_params.g, dt);
		apply_friction(h_e, tol_h, tol_s, qx_e, qy_e, sim_params.manning, sim_params.g, dt);
		apply_friction(h_s, tol_h, tol_s, qx_s, qy_s, sim_params.manning, sim_params.g, dt);
		apply_friction(h_w, tol_h, tol_s, qx_w, qy_w, sim_params.manning, sim_params.g, dt); 
		
		apply_friction(h_gauss_lower_x,   tol_h, tol_s, qx_gauss_lower_x,   qy_gauss_lower_x,   n, sim_params.g, dt);
		apply_friction(h_gauss_lower_x_n, tol_h, tol_s, qx_gauss_lower_x_n, qy_gauss_lower_x_n, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_x_e, tol_h, tol_s, qx_gauss_lower_x_e, qy_gauss_lower_x_e, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_x_s, tol_h, tol_s, qx_gauss_lower_x_s, qy_gauss_lower_x_s, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_x_w, tol_h, tol_s, qx_gauss_lower_x_w, qy_gauss_lower_x_w, n, sim_params.g, dt);
		
		apply_friction(h_gauss_upper_x,   tol_h, tol_s, qx_gauss_upper_x,   qy_gauss_upper_x,   n, sim_params.g, dt);
		apply_friction(h_gauss_upper_x_n, tol_h, tol_s, qx_gauss_upper_x_n, qy_gauss_upper_x_n, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_x_e, tol_h, tol_s, qx_gauss_upper_x_e, qy_gauss_upper_x_e, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_x_s, tol_h, tol_s, qx_gauss_upper_x_s, qy_gauss_upper_x_s, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_x_w, tol_h, tol_s, qx_gauss_upper_x_w, qy_gauss_upper_x_w, n, sim_params.g, dt);
		
		apply_friction(h_gauss_lower_y,   tol_h, tol_s, qx_gauss_lower_y,   qy_gauss_lower_y,   n, sim_params.g, dt);
		apply_friction(h_gauss_lower_y_n, tol_h, tol_s, qx_gauss_lower_y_n, qy_gauss_lower_y_n, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_y_e, tol_h, tol_s, qx_gauss_lower_y_e, qy_gauss_lower_y_e, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_y_s, tol_h, tol_s, qx_gauss_lower_y_s, qy_gauss_lower_y_s, n, sim_params.g, dt);
		apply_friction(h_gauss_lower_y_w, tol_h, tol_s, qx_gauss_lower_y_w, qy_gauss_lower_y_w, n, sim_params.g, dt);
		
		apply_friction(h_gauss_upper_y,   tol_h, tol_s, qx_gauss_upper_y,   qy_gauss_upper_y,   n, sim_params.g, dt);
		apply_friction(h_gauss_upper_y_n, tol_h, tol_s, qx_gauss_upper_y_n, qy_gauss_upper_y_n, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_y_e, tol_h, tol_s, qx_gauss_upper_y_e, qy_gauss_upper_y_e, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_y_s, tol_h, tol_s, qx_gauss_upper_y_s, qy_gauss_upper_y_s, n, sim_params.g, dt);
		apply_friction(h_gauss_upper_y_w, tol_h, tol_s, qx_gauss_upper_y_w, qy_gauss_upper_y_w, n, sim_params.g, dt);

		qx1x   = C(0.5) * (qx_gauss_upper_x   - qx_gauss_lower_x);
		qx1x_n = C(0.5) * (qx_gauss_upper_x_n - qx_gauss_lower_x_n);
		qx1x_e = C(0.5) * (qx_gauss_upper_x_e - qx_gauss_lower_x_e);
		qx1x_s = C(0.5) * (qx_gauss_upper_x_s - qx_gauss_lower_x_s);
		qx1x_w = C(0.5) * (qx_gauss_upper_x_w - qx_gauss_lower_x_w);
		
		qx1y   = C(0.5) * (qx_gauss_upper_y   - qx_gauss_lower_y);
		qx1y_n = C(0.5) * (qx_gauss_upper_y_n - qx_gauss_lower_y_n);
		qx1y_e = C(0.5) * (qx_gauss_upper_y_e - qx_gauss_lower_y_e);
		qx1y_s = C(0.5) * (qx_gauss_upper_y_s - qx_gauss_lower_y_s);
		qx1y_w = C(0.5) * (qx_gauss_upper_y_w - qx_gauss_lower_y_w);
		
		qy1x   = C(0.5) * (qy_gauss_upper_x   - qy_gauss_lower_x);
		qy1x_n = C(0.5) * (qy_gauss_upper_x_n - qy_gauss_lower_x_n);
		qy1x_e = C(0.5) * (qy_gauss_upper_x_e - qy_gauss_lower_x_e);
		qy1x_s = C(0.5) * (qy_gauss_upper_x_s - qy_gauss_lower_x_s);
		qy1x_w = C(0.5) * (qy_gauss_upper_x_w - qy_gauss_lower_x_w);

		qy1y   = C(0.5) * (qy_gauss_upper_y   - qy_gauss_lower_y);
		qy1y_n = C(0.5) * (qy_gauss_upper_y_n - qy_gauss_lower_y_n);
		qy1y_e = C(0.5) * (qy_gauss_upper_y_e - qy_gauss_lower_y_e);
		qy1y_s = C(0.5) * (qy_gauss_upper_y_s - qy_gauss_lower_y_s);
		qy1y_w = C(0.5) * (qy_gauss_upper_y_w - qy_gauss_lower_y_w);
	}
}