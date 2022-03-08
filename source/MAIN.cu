// RC's suggestion to manage Intellisense
#ifdef __INTELLISENSE__
    #ifndef __CUDACC__
        #define __CUDACC__
    #endif
#endif

//-----------------Custom headers-----------------//

// Kernels
#include "generate_all_morton_codes.cuh"
#include "copy_finest_coefficients.cuh"
#include "insert_point_srcs.cuh"
#include "reinsert_point_srcs.cuh"
#include "init_sig_details.cuh"
#include "zero_details.cuh"
#include "traverse_tree_of_sig_details.cuh"
#include "find_neighbours.cuh"
#include "get_compaction_flags.cuh"
#include "load_soln_and_nghbr_coeffs.cuh"
#include "add_ghost_cells.cuh"
#include "friction_implicit.cuh"
#include "fv1_update.cuh"
#include "dg2_update.cuh"

// Kernel wrappers
#include "get_nodal_values.cuh"
#include "get_modal_values.cuh"
#include "sort_finest_scale_coeffs_z_order.cuh"
#include "get_max_scale_coeffs.cuh"
#include "preflag_topo.cuh"
#include "encoding_all.cuh"
#include "get_reg_tree.cuh"
#include "decoding_all.cuh"
#include "rev_z_order_act_idcs.cuh"
#include "rev_z_order_reals.cuh"
#include "sort_neighbours_z_order.cuh"
#include "compaction.cuh"
#include "get_dt_CFL.cuh"

// Input/output
#include "read_bound_conds.h"
#include "read_cell_size.h"
#include "read_plot_params.h"
#include "read_respath.h"
#include "read_save_interval.h"
#include "read_sim_params.h"
#include "read_solver_params.h"
#include "read_test_case.h"
#include "write_all_raster_maps.cuh"
#include "write_c_prop_data.cuh"
#include "write_mesh_info.h"
#include "write_gauge_point_data.cuh"
#include "write_soln_planar.cuh"
#include "write_soln_row_major.cuh"
#include "write_soln_vtk.cuh"

// Helper functions
#include "get_lvl_idx.cuh"
#include "preflag_details.cuh"
#include "project_assem_sol.cuh"
#include "copy_to_buf_assem_sol.cuh"

// Sorting
#include "get_sorting_indices.cuh"

//------------------------------------------------//

/*
 * SYNTHETIC TEST CASES:
 * 
 * The following test cases can be run by inputting a number between 1 and 22 in inputs.par:
 * 
 * 1:  Wet 1D c-property x direction
 * 2:  Wet 1D c-property y direction
 * 3:  Wet-dry 1D c-property x direction
 * 4:  Wet-dry 1D c-property y direction
 * 5:  Wet dam break in x direction
 * 6:  Wet dam break in y direction
 * 7:  Dry dam break in x direction
 * 8:  Dry dam break in y direction
 * 9:  Dry dam break in x direction with friction
 * 10: Dry dam break in y direction with friction
 * 11: Wet building overtopping in x direction
 * 12: Wet building overtopping in y direction
 * 13: Wet-dry building overtopping in x direction
 * 14: Wet-dry building overtopping in y direction
 * 15: Triangular dam break in x direction
 * 16: Triangular dam break in y direction
 * 17: Parabolic bowl in x direction
 * 18: Parabolic bowl in y direction
 * 19: Three cones
 * 20: Differentiable blocks
 * 21: Non-differentiable blocks
 * 22: Radial dam break
 * 
 */

/*
 * TOPOGRAPHY AND FLOW VARIABLES:
 *
 *   z : topography
 *   h : water height
 * eta : free surface height (h + z)
 *  qx : discharge in x direction
 *  qy : discharge in y direction
 * 
 */


/*
 * STENCIL FOR MULTIRESOLUTION ANALYSIS (MRA):
 *
 * Fig 2 of Kesserwani and Sharifian et al. (2020) takes the origin to be the bottom left corner.
 * However, the origin in this code is taken to be the top left corner.
 * This means that numbering of the child sub-elements, or children, is flipped vertically, giving:
 *
 * |-----|-----|
 * |  0  |  1  |
 * |-----|-----|
 * |  2  |  3  |
 * |-----|-----|
 *
 */


/*
 * ARRAY OF HIERARCHY OF GRIDS:
 * 
 * As a result of MRA, the 2D mesh, or grid, is square i.e. the mesh dimensions are equal.
 * Furthermore, instead of a single grid, there is a hierarchy of grids that are stacked on top of one another.
 * The resolution of each grid becomes methodically finer the further up in the hierarchy.
 * The top-most grid is at the finest resolution, which is dictated by the maximum refinement level 'L'.
 * It consists of 4^L non-overlapping elements, resulting in a mesh dimension of 2^L elements.
 * Generally, at a given refinement level n, there are 4^n elements, where n = 0, 1, ..., L - 1, L.
 * Hence, in the entire hierarchy, there 4^0 + 4^l + ... + 4^L = (4^(L+1) - 1) / 3 elements.
 * The hierarchy of grids is thus stored in a 1D array of length (4^(L+1) - 1) / 3.
 * Each grid is transformed into a 1D structure by mapping its elements to a z-order curve.
 * The z-order curve of a grid is obtained by calculating and then sorting the Morton codes of its elements.
 * Within the 1D array, the z-order curve of a grid at n + 1 begins after that of a grid at n ends.
 * 
 */

// ======================================================================================================== //
// =============================================MAIN PROGRAM=============================================== //
// ======================================================================================================== //

int main
(
	int    argc,
	char **argv
)
{
	// begin timing from the beginning, as input is automated
	const clock_t start = clock();

	// ================ //
	// TEST CASE SET UP //
	// ================ //

	const char* input_filename = argv[1];
	
	const int test_case = read_test_case(input_filename);
	
	char respath[255] = {'\0'};
	read_respath(input_filename, respath);

	// ================ //

	// =========================================================== //
	// INITIALISATION OF VARIABLES AND INSTANTIATION OF STRUCTURES //
	// =========================================================== //

	// Structures setting up simulation
	SolverParams     solver_params = read_solver_params(input_filename);
	SimulationParams sim_params    = read_sim_params(test_case, input_filename, solver_params);
	PlottingParams   plot_params   = read_plot_params(input_filename);
	Depths1D         bcs           = read_bound_conds(test_case);
	SaveInterval     saveint       = read_save_interval(input_filename, "saveint");
	SaveInterval     massint       = read_save_interval(input_filename, "massint");

	// Variables
	int mesh_dim      = 1 << solver_params.L;
	int interface_dim = mesh_dim + 1;

	real dx_finest = (test_case != 0) ? (sim_params.xmax - sim_params.xmin) / mesh_dim : read_cell_size(input_filename);
	real dy_finest = (test_case != 0) ? (sim_params.ymax - sim_params.ymin) / mesh_dim : read_cell_size(input_filename);
	real dt        = C(0.001);

	int num_finest_elems      = mesh_dim * mesh_dim;
	int num_blocks_finest     = get_num_blocks(num_finest_elems, THREADS_PER_BLOCK);
	int num_threads_traversal = num_finest_elems / 4;
	int num_blocks_traversal  = get_num_blocks(num_threads_traversal, THREADS_PER_BLOCK);
	int num_all_elems         = get_lvl_idx(solver_params.L + 1);
	int num_details           = get_lvl_idx(solver_params.L);
	int num_blocks_details    = get_num_blocks(num_details, THREADS_PER_BLOCK);
	int num_blocks_sol        = 0;
	int num_blocks_all        = get_num_blocks(num_all_elems, THREADS_PER_BLOCK);
	
	HierarchyIndex finest_lvl_idx = get_lvl_idx(solver_params.L);
	
	// Structures
	Maxes maxes = { C(1.0), C(1.0), C(1.0), C(1.0) };
	
	GaugePoints  gauge_points (input_filename, sim_params, dx_finest);
	Boundaries   boundaries   (input_filename, sim_params, dx_finest, test_case);
	PointSources point_sources(input_filename, sim_params, dx_finest, test_case, dt);
	
	clock_t end             = clock();
	real    run_time        = C(0.0);
	real    time_now        = C(0.0);
	bool    first_t_step    = true;
	bool    for_nghbrs      = false;
	bool    rkdg2           = false;
	float   avg_cuda_time   = 0.0f;
	int     steps           = 0;
	real    compression     = C(0.0);

	NodalValues       d_nodal_vals      (interface_dim);
	AssembledSolution d_assem_sol       (num_finest_elems, solver_params.solver_type);
	AssembledSolution d_buf_assem_sol   (num_finest_elems, solver_params.solver_type);
	AssembledSolution d_plot_assem_sol  (num_finest_elems, solver_params.solver_type);
	Neighbours        d_neighbours      (num_finest_elems, solver_params.solver_type);
	Neighbours        d_buf_neighbours  (num_finest_elems, solver_params.solver_type);
	ScaleCoefficients d_scale_coeffs    (num_all_elems,    solver_params.solver_type);
	Details           d_details         (num_details,      solver_params.solver_type);
	CompactionFlags   d_compaction_flags(num_finest_elems);
	FinestGrid        p_finest_grid     (num_finest_elems);
	
	// Bytesizes
	size_t bytes_morton  = num_finest_elems * sizeof(MortonCode);
	size_t bytes_details = num_details      * sizeof(real);
	size_t bytes_soln    = num_finest_elems * sizeof(real);

	// Arrays
	MortonCode* d_morton_codes        = (MortonCode*)malloc_device(bytes_morton);
	MortonCode* d_sorted_morton_codes = (MortonCode*)malloc_device(bytes_morton);
	MortonCode* d_indices             = (MortonCode*)malloc_device(bytes_morton);
	MortonCode* d_rev_z_order         = (MortonCode*)malloc_device(bytes_morton);
	MortonCode* d_rev_row_major       = (MortonCode*)malloc_device(bytes_morton);
	real*       d_eta_temp            = (real*)malloc_device(bytes_soln);
	real*       d_norm_details        = (real*)malloc_device(bytes_details);
	bool*       d_sig_details         = (bool*)malloc_device(num_details);
	real*       d_dt_CFL              = (real*)malloc_device(bytes_soln);
	
	bool* d_preflagged_details = preflag_details
	(
		boundaries, 
		point_sources, 
		gauge_points, 
		sim_params, 
		num_details, 
		solver_params.L, 
		test_case
	);

	// =========================================================== //

	// ================ //
	// INPUT AND OUTPUT //
	// ================ //

	write_mesh_info(sim_params, mesh_dim, respath);

	// ================ //

	/*
		
		cudaEvent_t cuda_begin, cuda_end;
		cudaEventCreate(&cuda_begin);
		cudaEventCreate(&cuda_end);

		cudaEventRecord(cuda_begin);


		cudaEventRecord(cuda_end);
		cudaEventSynchronize(cuda_end);

		float cuda_time = 0;
		cudaEventElapsedTime(&cuda_time, cuda_begin, cuda_end);
		cudaEventDestroy(cuda_begin);
		cudaEventDestroy(cuda_end);

		avg_cuda_time += cuda_time;

	*/

	// ================================ //
	// PREPROCESSING BEFORE SOLVER LOOP //
	// ================================ //
	
	if (test_case != 0)
	{
		get_nodal_values
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

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	get_modal_values
	(
		d_nodal_vals,
		d_buf_assem_sol,
		solver_params,
		sim_params,
		mesh_dim,
		interface_dim,
		test_case,
		input_filename
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());
	
	write_all_raster_maps
	(
		respath,
		d_buf_assem_sol,
		sim_params,
		solver_params,
		massint,
		mesh_dim,
		dx_finest,
		first_t_step
	);
	
	generate_all_morton_codes<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_morton_codes,
		d_indices,
		mesh_dim
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	get_sorting_indices
	(
		d_morton_codes,
		d_sorted_morton_codes,
		d_buf_assem_sol,
		d_assem_sol,
		d_indices,
		d_rev_z_order,
		d_rev_row_major,
		solver_params
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	sort_finest_scale_coeffs_z_order<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_buf_assem_sol,
		d_assem_sol,
		d_rev_z_order,
		solver_params
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	copy_finest_coefficients<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_assem_sol,
		d_scale_coeffs,
		solver_params,
		finest_lvl_idx
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	if (point_sources.num_srcs > 0)
	{
		insert_point_srcs<<<get_num_blocks(point_sources.num_srcs, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
		(
			d_assem_sol, 
			point_sources, 
			dt, 
			dx_finest
		);
	}

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	init_sig_details<<<num_blocks_details, THREADS_PER_BLOCK>>> //d_sig_details[idx] = true;
	(
		d_sig_details, 
		num_details
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	maxes = get_max_scale_coeffs(d_assem_sol, d_eta_temp);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	preflag_topo
	(
		d_scale_coeffs, 
		d_details,  
		d_preflagged_details, 
		maxes,
		solver_params,
		sim_params,
		first_t_step
	);

	CHECK_CUDA_ERROR(peek());
	CHECK_CUDA_ERROR(sync());

	// ================================ //

	// ================ //
	// MAIN SOLVER LOOP //
	// ================ //
	
	while (time_now < sim_params.time)
	{
		time_now += dt;

		if ( (time_now - sim_params.time) > C(0.0) )
		{
			time_now -= dt;
			dt = sim_params.time - time_now;
			time_now += dt;
		}
		
		zero_details<<<num_blocks_details, THREADS_PER_BLOCK>>>
		(
			d_details,
			d_norm_details,
			num_details,
			solver_params
		);

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());

		maxes = get_max_scale_coeffs(d_assem_sol, d_eta_temp);

		if (!first_t_step)
		{						
			reinsert_assem_sol<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_assem_sol,
				d_assem_sol.act_idcs,
				d_scale_coeffs,
				solver_params
			);
		}

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());

		point_sources.update_all_srcs(time_now);

		if (point_sources.num_srcs > 0)
		{
			reinsert_point_srcs<<<get_num_blocks(point_sources.num_srcs, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
			(
				d_scale_coeffs, 
				point_sources, 
				dt, 
				dx_finest, 
				solver_params.L
			);
		}

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());

		if (solver_params.epsilon > C(0.0) || first_t_step)
		{
		    for_nghbrs = false;
		    
		    encoding_all
		    (
		    	d_scale_coeffs,
		    	d_details,
		    	d_norm_details,
		    	d_sig_details,
		    	d_preflagged_details,
		    	maxes,
		    	solver_params,
		    	for_nghbrs
		    );
		    
		    get_reg_tree
		    (
		    	d_sig_details,
		    	solver_params
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    decoding_all // contains extra sig
		    (
		    	d_sig_details,
		    	d_norm_details,
		    	d_details,
		    	d_scale_coeffs,
		    	solver_params
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    traverse_tree_of_sig_details<<<num_blocks_traversal, THREADS_PER_BLOCK>>>
		    (
		    	d_sig_details,
		    	d_scale_coeffs,
		    	d_buf_assem_sol,
		    	num_threads_traversal,
		    	solver_params
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    rev_z_order_act_idcs<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
				d_rev_row_major,
		    	d_buf_assem_sol,
		    	d_assem_sol,
		    	num_finest_elems
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    find_neighbours<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_assem_sol,
		    	d_neighbours,
		    	sim_params,
		    	mesh_dim
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    get_compaction_flags<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_buf_assem_sol,
		    	d_compaction_flags,
		    	num_finest_elems
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    sort_neighbours_z_order<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_neighbours,
		    	d_buf_neighbours,
		    	d_rev_z_order,
		    	num_finest_elems,
		    	solver_params
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		    
		    compaction
		    (
		    	d_buf_assem_sol,
		    	d_assem_sol,
		    	d_buf_neighbours,
		    	d_neighbours,
		    	d_compaction_flags,
		    	num_finest_elems,
		    	solver_params
		    );
		    
		    CHECK_CUDA_ERROR(peek());
		    CHECK_CUDA_ERROR(sync());
		}

		// GRID DIMENSIONS BASED ON ASSEMBLED SOLUTION LENGTH //

		num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);

		// -------------------------------------------------- //

		load_soln_and_nghbr_coeffs<<<num_blocks_sol, THREADS_PER_BLOCK>>>
		(
			d_neighbours,
			d_scale_coeffs,
			d_assem_sol,
			solver_params
		);

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());
		
		boundaries.update_all_inlets(input_filename, time_now);

		add_ghost_cells<<<num_blocks_sol, THREADS_PER_BLOCK>>>
		(
			d_assem_sol,
			d_neighbours,
			solver_params,
			sim_params,
			boundaries,
			dt,
			dx_finest,
			test_case
		);

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());

		if ( sim_params.manning > C(0.0) )
		{
			friction_implicit<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_assem_sol,
				d_neighbours,
				solver_params, 
				sim_params, 
				dt
			);
		}

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());
		
		if (solver_params.solver_type == HWFV1)
		{
			fv1_update<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_neighbours,
				d_assem_sol,
				solver_params,
				sim_params,
				dx_finest,
				dy_finest,
				dt,
				d_dt_CFL
			);
		}
		else if (solver_params.solver_type == MWDG2)
		{
			copy_to_buf_assem_sol
			(
				d_assem_sol, 
				d_buf_assem_sol
			);

			rkdg2 = false;

			dg2_update<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_neighbours, 
				d_assem_sol, 
				d_buf_assem_sol, 
				solver_params, 
				sim_params, 
				dx_finest, 
				dy_finest, 
				dt, 
				test_case,
				d_dt_CFL,
				rkdg2
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());
			
			reinsert_assem_sol<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_buf_assem_sol,
				d_assem_sol.act_idcs,
				d_scale_coeffs,
				solver_params
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());
			
			for_nghbrs = true;

			encoding_all
			(
				d_scale_coeffs,
				d_details,
				d_norm_details,
				d_sig_details,
				d_preflagged_details,
				maxes,
				solver_params,
				for_nghbrs
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());

			load_soln_and_nghbr_coeffs<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_neighbours,
				d_scale_coeffs,
				d_buf_assem_sol,
				solver_params
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());

			add_ghost_cells<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_buf_assem_sol,
				d_neighbours,
				solver_params,
				sim_params,
				boundaries,
				dt,
				dx_finest,
				test_case
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());

			rkdg2 = true;

			dg2_update<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_neighbours, 
				d_buf_assem_sol, 
				d_assem_sol, 
				solver_params, 
				sim_params, 
				dx_finest, 
				dy_finest, 
				dt, 
				test_case, 
				d_dt_CFL,
				rkdg2
			);

			CHECK_CUDA_ERROR(peek());
			CHECK_CUDA_ERROR(sync());
		}

		dt = get_dt_CFL(d_dt_CFL, d_assem_sol.length);

		CHECK_CUDA_ERROR(peek());
		CHECK_CUDA_ERROR(sync());

		// --------------------------------------------- //
		// -------------- WRITING TO FILE -------------- //
		// --------------------------------------------- //

		if ( saveint.save(time_now) )
		{
			project_assem_sol
			(
				mesh_dim,
				d_sig_details,
				d_scale_coeffs,
				d_buf_assem_sol,
				solver_params,
				d_rev_z_order,
				d_indices,
				d_assem_sol,
				d_plot_assem_sol
			);
			
			if (plot_params.row_major)
			{
				write_soln_row_major
				(
					respath,
					mesh_dim,
					d_sig_details,
					d_scale_coeffs,
					d_buf_assem_sol,
					solver_params,
					d_rev_z_order,
					d_indices,
					d_assem_sol,
					d_plot_assem_sol,
					saveint
				);
			}

			if (plot_params.vtk)
			{
				write_soln_vtk
				(
					respath,
					d_assem_sol,
					d_dt_CFL,
					dx_finest,
					dy_finest,
					sim_params,
					solver_params,
					saveint
				);
			}
			
			if (plot_params.planar)
			{
				write_soln_planar
				(
					respath,
					d_assem_sol,
					dx_finest,
					dy_finest,
					sim_params,
					solver_params,
					saveint
				);
			}

			if (plot_params.raster_out)
			{
				write_all_raster_maps
				(
					respath,
					d_plot_assem_sol,
					sim_params,
					solver_params,
					saveint,
					mesh_dim,
					dx_finest,
					first_t_step
				);
			}

			if (plot_params.c_prop)
			{
				write_c_prop_data
				(
					respath,
					start,
					solver_params,
					d_assem_sol,
					time_now,
					first_t_step
				);
			}
		}

		if ( massint.save(time_now) )
		{
			if (plot_params.cumulative)
			{
			    write_cumu_sim_time
			    (
			        start,
			        time_now,
			        respath,
			        first_t_step
			    );
			}
			
			project_assem_sol
			(
				mesh_dim,
				d_sig_details,
				d_scale_coeffs,
				d_buf_assem_sol,
				solver_params,
				d_rev_z_order,
				d_indices,
				d_assem_sol,
				d_plot_assem_sol
			);
			
			write_gauge_point_data
			(
				respath,
				mesh_dim,
				d_sig_details,
				d_scale_coeffs,
				d_buf_assem_sol,
				solver_params,
				plot_params,
				d_rev_z_order,
				d_indices,
				d_assem_sol,
				d_plot_assem_sol,
				p_finest_grid,
				gauge_points,
				time_now,
				first_t_step
			);
		}

		// --------------------------------------------- //
		// --------------------------------------------- //
		// --------------------------------------------- //

		compression = C(100.0) - C(100.0) * d_assem_sol.length / (sim_params.xsz * sim_params.ysz);

		//printf
		(
			"Elements: %d, compression: %f%%, time step: %.15f, steps: %d, sim time: %f\n", 
			d_assem_sol.length, compression, dt, ++steps, time_now
		);
		
 		first_t_step = false;
	}

	end = clock();

	run_time = (real)(end - start) / CLOCKS_PER_SEC;
	
	printf("Loop time: %f s\n", run_time);

	printf("Average time step: %f s\n", sim_params.time / steps);
	printf("Average kernel time: %f ms\n", avg_cuda_time);

	// =================== //
	// DEALLOCATING MEMORY //
	// =================== //

	CHECK_CUDA_ERROR( free_device(d_morton_codes) );
	CHECK_CUDA_ERROR( free_device(d_sorted_morton_codes) );
	CHECK_CUDA_ERROR( free_device(d_indices) );
	CHECK_CUDA_ERROR( free_device(d_rev_z_order) );
	CHECK_CUDA_ERROR( free_device(d_rev_row_major) );
	CHECK_CUDA_ERROR( free_device(d_eta_temp) );
	CHECK_CUDA_ERROR( free_device(d_sig_details) );
	CHECK_CUDA_ERROR( free_device(d_preflagged_details) );
	CHECK_CUDA_ERROR( free_device(d_norm_details) );
	CHECK_CUDA_ERROR( free_device(d_dt_CFL) );
	
	//reset();

	// =================== //

    return 0;
}

//==========================================================================================================//