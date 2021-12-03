#include "init_nodal_values.cuh"

/*
 * FLATTENED ARRAY RATIONALE:
 *
 * 1D arrays emulating 2D arrays are used to hold the flow variables at the finest resolution.
 * These 1D arrays are obtained by 'flattening' a 2D array.
 * The following is an example of a 2D array of dimensions 3x3, that uses 'row-major indexing':
 *
 * ---|---|---
 * -0-|-1-|-2-
 * ---|---|---
 * -3-|-4-|-5-
 * ---|---|---
 * -6-|-7-|-8-
 * ---|---|---
 *
 * It can be flattened into a 1D array of length 3x3 = 9:
 *
 * ---|---|---|---|---|---|---|---|---
 * -0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-
 * ---|---|---|---|---|---|---|---|---
 *
 * As example, consider a 2D array that has 'num_cols' columns flattened into a 1D array.
 * To access the nth element of the the mth row, an index for the 1D array needs to be obtained.
 * This index = (m - 1) * num_cols + (n - 1), -1 since indexing starts from zero.
 * The reverse process can also be done i.e. finding the row and column given the 1D index and the num_cols:
 * - row number    = index / num_cols
 * - column number = index % num_cols
 */

__global__
void init_nodal_values
(
	NodalValues          d_nodal_vals,
	real                 dx_finest,
	real                 dy_finest,
	Depths1D             bcs,
	SimulationParameters sim_params,
	int                  interface_dim,
	int                  test_case
)
{
	HierarchyIndex idx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( idx >= (interface_dim * interface_dim) ) return;

	// obtaining 2D indices from the 'flattened' index (here 'idx')
	Coordinate x_idx = idx % interface_dim;
	Coordinate y_idx = idx / interface_dim;

	// calculating x- and y-nodal values
	const real x_int = sim_params.xmin + dx_finest * x_idx;
	const real y_int = sim_params.ymin + dy_finest * y_idx;

	real& z_int = d_nodal_vals.z[idx];

	z_int = bed_data
	(
		x_int, 
		y_int, 
		test_case
	);

	d_nodal_vals.h[idx] = h_init
	(
		x_int, 
		y_int, 
		z_int, 
		bcs, 
		test_case
	);

	d_nodal_vals.qx[idx] = 0;
	d_nodal_vals.qy[idx] = 0;
}