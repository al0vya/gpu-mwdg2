#include "get_nodal_values.cuh"

__host__ void get_nodal_values
(
	NodalValues&                d_nodal_vals,
	const real&                 dx_finest,
	const real&                 dy_finest,
	const Depths1D&             bcs,
	const SimulationParameters& sim_params,
	const int&                  interface_dim,
	const int&                  test_case
)
{
	// number of nodes
	int num_elements = interface_dim * interface_dim;
	int num_blocks   = get_num_blocks(num_elements, THREADS_PER_BLOCK);

	init_nodal_values<<<num_blocks, THREADS_PER_BLOCK>>>
	(
		d_nodal_vals,
		dx_finest,
		dy_finest,
		bcs,
		sim_params,
		interface_dim,
		test_case
	);
}