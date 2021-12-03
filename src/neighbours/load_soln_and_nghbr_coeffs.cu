#include "load_soln_and_nghbr_coeffs.cuh"

__global__
void load_soln_and_nghbr_coeffs
(
	Neighbours        d_neighbours,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_assem_sol,
	SolverParameters  solver_params
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	HierarchyIndex soln  = d_assem_sol.act_idcs[idx];
	HierarchyIndex north = d_neighbours.north.act_idcs[idx];
	HierarchyIndex east  = d_neighbours.east.act_idcs[idx];
	HierarchyIndex south = d_neighbours.south.act_idcs[idx];
	HierarchyIndex west  = d_neighbours.west.act_idcs[idx];

	north = (north == -1) ? 0 : north;
	east  = (east  == -1) ? 0 : east;
	south = (south == -1) ? 0 : south;
	west  = (west  == -1) ? 0 : west;

	d_assem_sol.h0[idx]  = d_scale_coeffs.eta0[soln] - d_scale_coeffs.z0[soln];
	d_assem_sol.qx0[idx] = d_scale_coeffs.qx0[soln];
	d_assem_sol.qy0[idx] = d_scale_coeffs.qy0[soln];
	d_assem_sol.z0[idx]  = d_scale_coeffs.z0[soln];

	d_neighbours.north.h0[idx]  = d_scale_coeffs.eta0[north] - d_scale_coeffs.z0[north];
	d_neighbours.north.qx0[idx] = d_scale_coeffs.qx0[north];
	d_neighbours.north.qy0[idx] = d_scale_coeffs.qy0[north];
	d_neighbours.north.z0[idx]  = d_scale_coeffs.z0[north];

	d_neighbours.east.h0[idx]  = d_scale_coeffs.eta0[east] - d_scale_coeffs.z0[east];
	d_neighbours.east.qx0[idx] = d_scale_coeffs.qx0[east];
	d_neighbours.east.qy0[idx] = d_scale_coeffs.qy0[east];
	d_neighbours.east.z0[idx]  = d_scale_coeffs.z0[east];

	d_neighbours.south.h0[idx]  = d_scale_coeffs.eta0[south] - d_scale_coeffs.z0[south];
	d_neighbours.south.qx0[idx] = d_scale_coeffs.qx0[south];
	d_neighbours.south.qy0[idx] = d_scale_coeffs.qy0[south];
	d_neighbours.south.z0[idx]  = d_scale_coeffs.z0[south];

	d_neighbours.west.h0[idx]  = d_scale_coeffs.eta0[west] - d_scale_coeffs.z0[west];
	d_neighbours.west.qx0[idx] = d_scale_coeffs.qx0[west];
	d_neighbours.west.qy0[idx] = d_scale_coeffs.qy0[west];
	d_neighbours.west.z0[idx]  = d_scale_coeffs.z0[west];

	if (solver_params.solver_type == MWDG2)
	{
		d_assem_sol.h1x[idx]  = d_scale_coeffs.eta1x[soln] - d_scale_coeffs.z1x[soln];
		d_assem_sol.qx1x[idx] = d_scale_coeffs.qx1x[soln];
		d_assem_sol.qy1x[idx] = d_scale_coeffs.qy1x[soln];
		d_assem_sol.z1x[idx]  = d_scale_coeffs.z1x[soln];

		d_neighbours.north.h1x[idx]  = d_scale_coeffs.eta1x[north] - d_scale_coeffs.z1x[north];
		d_neighbours.north.qx1x[idx] = d_scale_coeffs.qx1x[north];
		d_neighbours.north.qy1x[idx] = d_scale_coeffs.qy1x[north];
		d_neighbours.north.z1x[idx]  = d_scale_coeffs.z1x[north];

		d_neighbours.east.h1x[idx]  = d_scale_coeffs.eta1x[east] - d_scale_coeffs.z1x[east];
		d_neighbours.east.qx1x[idx] = d_scale_coeffs.qx1x[east];
		d_neighbours.east.qy1x[idx] = d_scale_coeffs.qy1x[east];
		d_neighbours.east.z1x[idx]  = d_scale_coeffs.z1x[east];

		d_neighbours.south.h1x[idx]  = d_scale_coeffs.eta1x[south] - d_scale_coeffs.z1x[south];
		d_neighbours.south.qx1x[idx] = d_scale_coeffs.qx1x[south];
		d_neighbours.south.qy1x[idx] = d_scale_coeffs.qy1x[south];
		d_neighbours.south.z1x[idx]  = d_scale_coeffs.z1x[south];

		d_neighbours.west.h1x[idx]  = d_scale_coeffs.eta1x[west] - d_scale_coeffs.z1x[west];
		d_neighbours.west.qx1x[idx] = d_scale_coeffs.qx1x[west];
		d_neighbours.west.qy1x[idx] = d_scale_coeffs.qy1x[west];
		d_neighbours.west.z1x[idx]  = d_scale_coeffs.z1x[west];

		d_assem_sol.h1y[idx]  = d_scale_coeffs.eta1y[soln] - d_scale_coeffs.z1y[soln];
		d_assem_sol.qx1y[idx] = d_scale_coeffs.qx1y[soln];
		d_assem_sol.qy1y[idx] = d_scale_coeffs.qy1y[soln];
		d_assem_sol.z1y[idx]  = d_scale_coeffs.z1y[soln];

		d_neighbours.north.h1y[idx]  = d_scale_coeffs.eta1y[north] - d_scale_coeffs.z1y[north];
		d_neighbours.north.qx1y[idx] = d_scale_coeffs.qx1y[north];
		d_neighbours.north.qy1y[idx] = d_scale_coeffs.qy1y[north];
		d_neighbours.north.z1y[idx]  = d_scale_coeffs.z1y[north];

		d_neighbours.east.h1y[idx]  = d_scale_coeffs.eta1y[east] - d_scale_coeffs.z1y[east];
		d_neighbours.east.qx1y[idx] = d_scale_coeffs.qx1y[east];
		d_neighbours.east.qy1y[idx] = d_scale_coeffs.qy1y[east];
		d_neighbours.east.z1y[idx]  = d_scale_coeffs.z1y[east];

		d_neighbours.south.h1y[idx]  = d_scale_coeffs.eta1y[south] - d_scale_coeffs.z1y[south];
		d_neighbours.south.qx1y[idx] = d_scale_coeffs.qx1y[south];
		d_neighbours.south.qy1y[idx] = d_scale_coeffs.qy1y[south];
		d_neighbours.south.z1y[idx]  = d_scale_coeffs.z1y[south];

		d_neighbours.west.h1y[idx]  = d_scale_coeffs.eta1y[west] - d_scale_coeffs.z1y[west];
		d_neighbours.west.qx1y[idx] = d_scale_coeffs.qx1y[west];
		d_neighbours.west.qy1y[idx] = d_scale_coeffs.qy1y[west];
		d_neighbours.west.z1y[idx]  = d_scale_coeffs.z1y[west];
	}
}