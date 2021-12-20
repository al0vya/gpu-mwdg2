#include "modal_projections.cuh"

/*
 * SQUARE SHARED MEMORY BLOCKS:
 * 
 * The modal projections are done using a kernel that uses square blocks of shared memory.
 * Only square blocks fit neatly into the square mesh (diagram below), and also minimse global memory accesses.
 *
 * -----|-----|-----|-----
 * -0,0-|-0,1-|-0,2-|-0,3-
 * -----|-----|-----|-----
 * -1,0-|-1,1-|-1,2-|-1,3-
 * -----|-----|-----|-----
 * -2,0-|-2,1-|-2,2-|-2,3-
 * -----|-----|-----|-----
 * -3,0-|-3,1-|-3,2-|-3,3-
 * -----|-----|-----|-----
 *
 * For example, (1,2) is the 3rd block in the 2nd row of blocks.
 * Each square block must be emulated using a 1D array, as done previously (grep 'FLATTENED ARRAY RATIONALE').
 * Since square blocks are used, the number of threads must be a square number i.e. 64, 256 or 1024.
 * The square root of this number is the side length of the emulated square block, #defined as 'SHARED_MEMORY_BLOCK_DIM'.
 * There can be (mesh_dim / SHARED_MEMORY_BLOCK_DIM) blocks per row.
 * This number and the side length are equal since the mesh and block of shared memory are both square.
 * Hence, it is stored as 'shared_memory_blocks_per_side'.
 */

__global__
void modal_projections
(
	NodalValues       d_nodal_vals,
	AssembledSolution d_assem_sol,
	SolverParams  solver_params,
	int               mesh_dim,
	int               interface_dim
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockDim.x * blockIdx.x + t_idx;
	
	if ( idx >= d_assem_sol.length ) return;

	Coordinate x = idx % mesh_dim;
	Coordinate y = idx / mesh_dim;

	HierarchyIndex SW =  y      * interface_dim + x;
	HierarchyIndex SE =  y      * interface_dim + x + 1;
	HierarchyIndex NW = (y + 1) * interface_dim + x;
	HierarchyIndex NE = (y + 1) * interface_dim + x + 1;

	real h_SW = d_nodal_vals.h[SW];
	real h_SE = d_nodal_vals.h[SE];
	real h_NW = d_nodal_vals.h[NW];
	real h_NE = d_nodal_vals.h[NE];

	real qx_SW = d_nodal_vals.qx[SW];
	real qx_SE = d_nodal_vals.qx[SE];
	real qx_NW = d_nodal_vals.qx[NW];
	real qx_NE = d_nodal_vals.qx[NE];

	real qy_SW = d_nodal_vals.qy[SW];
	real qy_SE = d_nodal_vals.qy[SE];
	real qy_NW = d_nodal_vals.qy[NW];
	real qy_NE = d_nodal_vals.qy[NE];

	real z_SW = d_nodal_vals.z[SW];
	real z_SE = d_nodal_vals.z[SE];
	real z_NW = d_nodal_vals.z[NW];
	real z_NE = d_nodal_vals.z[NE];

	d_assem_sol.h0[idx]  = C(0.25) * (h_NW  + h_NE  + h_SE  + h_SW );
	d_assem_sol.qx0[idx] = C(0.25) * (qx_NW + qx_NE + qx_SE + qx_SW);
	d_assem_sol.qy0[idx] = C(0.25) * (qy_NW + qy_NE + qy_SE + qy_SW);
	d_assem_sol.z0[idx]  = C(0.25) * (z_NW  + z_NE  + z_SE  + z_SW );

	if (solver_params.solver_type == MWDG2)
	{
		d_assem_sol.h1x[idx]  = (h_NE  - h_NW  + h_SE  - h_SW ) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.qx1x[idx] = (qx_NE - qx_NW + qx_SE - qx_SW) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.qy1x[idx] = (qy_NE - qy_NW + qy_SE - qy_SW) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.z1x[idx]  = (z_NE  - z_NW  + z_SE  - z_SW ) / ( C(4.0) * sqrt( C(3.0) ) );

		d_assem_sol.h1y[idx]  = (h_NE  - h_SE  + h_NW  - h_SW ) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.qx1y[idx] = (qx_NE - qx_SE + qx_NW - qx_SW) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.qy1y[idx] = (qy_NE - qy_SE + qy_NW - qy_SW) / ( C(4.0) * sqrt( C(3.0) ) );
		d_assem_sol.z1y[idx]  = (z_NE  - z_SE  + z_NW  - z_SW ) / ( C(4.0) * sqrt( C(3.0) ) );
	}
}