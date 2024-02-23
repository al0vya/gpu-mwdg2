#include "run_simulation.cuh"

void run_simulation
(
	int    argc,
	char** argv
)
{
	const clock_t start = clock();

	const char* input_filename = argv[argc - 1];
	
	const int test_case = read_test_case(input_filename);
	
	// Structures for setting up simulation
	SolverParams     solver_params(input_filename);
	SimulationParams sim_params(test_case, input_filename, solver_params.L);
	PlottingParams   plot_params(input_filename);
	Depths1D         bcs(test_case);
	SaveInterval     saveint(input_filename, "saveint");
	SaveInterval     massint(input_filename, "massint");

	read_command_line_params
	(
		argc,
		argv,
		sim_params,
		solver_params,
		plot_params
	);

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
	Maxes maxes = { C(1.0), C(1.0), C(1.0), C(1.0), C(1.0) };
	
	StagePoints  stage_points (input_filename, sim_params, dx_finest);
	Boundaries   boundaries   (input_filename, sim_params, dx_finest, test_case);
	PointSources point_sources(input_filename, sim_params, dx_finest, test_case, dt);
	
	clock_t end             = clock();
	clock_t mra_start       = clock();
	clock_t mra_end         = clock();
	clock_t solver_start    = clock();
	clock_t solver_end      = clock();
	real    run_time        = C(0.0);
	real    current_time    = C(0.0);
	real    time_mra        = C(0.0);
	real    time_solver     = C(0.0);
	bool    first_timestep  = true;
	bool    for_nghbrs      = false;
	bool    rkdg2           = false;
	float   avg_cuda_time   = 0.0f;
	int     timestep        = 1;
	real    compression     = C(0.0);

	NodalValues       d_nodal_vals      (interface_dim);
	AssembledSolution d_assem_sol       (num_finest_elems, solver_params.solver_type);
	AssembledSolution d_buf_assem_sol   (num_finest_elems, solver_params.solver_type);
	AssembledSolution d_plot_assem_sol  (num_finest_elems, solver_params.solver_type);
	Neighbours        d_neighbours      (num_finest_elems, solver_params.solver_type);
	Neighbours        d_buf_neighbours  (num_finest_elems, solver_params.solver_type);
	ScaleCoefficients d_scale_coeffs    (solver_params);
	Details           d_details         (solver_params);
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
		stage_points, 
		sim_params, 
		solver_params, 
		num_details, 
		solver_params.L, 
		test_case
	);

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

	generate_all_morton_codes<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_morton_codes,
		d_indices,
		mesh_dim
	);

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

	sort_finest_scale_coeffs_z_order<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_buf_assem_sol,
		d_assem_sol,
		d_rev_z_order,
		solver_params
	);

	copy_finest_coefficients<<<num_blocks_finest, THREADS_PER_BLOCK>>>
	(
		d_assem_sol,
		d_scale_coeffs,
		solver_params,
		finest_lvl_idx
	);

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

	init_sig_details<<<num_blocks_details, THREADS_PER_BLOCK>>> //d_sig_details[idx] = true;
	(
		d_sig_details, 
		num_details
	);

	maxes = get_max_scale_coeffs(d_assem_sol, d_eta_temp);

	#if _RUN_UNIT_TESTS
	generate_data_unit_test_preflag_topo
	(
		plot_params.dirroot,
		"input",
		d_scale_coeffs,
		d_details,
		d_preflagged_details,
		solver_params
	);
	#endif

	preflag_topo
	(
		d_scale_coeffs, 
		d_details,  
		d_preflagged_details, 
		maxes,
		solver_params,
		sim_params,
		first_timestep
	);

	#if _RUN_UNIT_TESTS
	generate_data_unit_test_preflag_topo
	(
		plot_params.dirroot,
		"output",
		d_scale_coeffs,
		d_details,
		d_preflagged_details,
		solver_params
	);
	#endif

	// main solver loop
	while (current_time < sim_params.time)
	{
		current_time += dt;

		if ( (current_time - sim_params.time) > C(0.0) )
		{
			current_time -= dt;
			dt = sim_params.time - current_time;
			current_time += dt;
		}
		
		mra_start = clock();

		zero_details<<<num_blocks_details, THREADS_PER_BLOCK>>>
		(
			d_details,
			d_norm_details,
			num_details,
			solver_params
		);

		maxes = get_max_scale_coeffs(d_assem_sol, d_eta_temp);

		if (!first_timestep)
		{						
			reinsert_assem_sol<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_assem_sol,
				d_assem_sol.act_idcs,
				d_scale_coeffs,
				solver_params
			);
		}

		point_sources.update_all_srcs(current_time);

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

		if (solver_params.epsilon > C(0.0) || first_timestep)
		{
		    for_nghbrs = false;
		    
			#if _RUN_UNIT_TESTS
			generate_data_unit_test_encoding_all
			(
				plot_params.dirroot,
				"input",
				d_scale_coeffs,
				d_details,
				d_norm_details,
				d_sig_details,
				d_preflagged_details,
				solver_params,
				timestep
			);
			#endif

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
		    
			#if _RUN_UNIT_TESTS
			generate_data_unit_test_encoding_all
			(
				plot_params.dirroot,
				"output",
				d_scale_coeffs,
				d_details,
				d_norm_details,
				d_sig_details,
				d_preflagged_details,
				solver_params,
				timestep
			);
			#endif

		    get_reg_tree
		    (
		    	d_sig_details,
		    	solver_params
		    );
		    
		    decoding_all // contains extra sig
		    (
		    	d_sig_details,
		    	d_norm_details,
		    	d_details,
		    	d_scale_coeffs,
		    	solver_params
		    );
		    
		    traverse_tree_of_sig_details<<<num_blocks_traversal, THREADS_PER_BLOCK>>>
		    (
		    	d_sig_details,
		    	d_scale_coeffs,
		    	d_buf_assem_sol,
		    	num_threads_traversal,
		    	solver_params
		    );
		    
		    rev_z_order_act_idcs<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
				d_rev_row_major,
		    	d_buf_assem_sol,
		    	d_assem_sol,
		    	num_finest_elems
		    );
		    
		    find_neighbours<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_assem_sol,
		    	d_neighbours,
		    	sim_params,
		    	mesh_dim
		    );
		    
		    get_compaction_flags<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_buf_assem_sol,
		    	d_compaction_flags,
		    	num_finest_elems
		    );
		    
		    sort_neighbours_z_order<<<num_blocks_finest, THREADS_PER_BLOCK>>>
		    (
		    	d_neighbours,
		    	d_buf_neighbours,
		    	d_rev_z_order,
		    	num_finest_elems,
		    	solver_params
		    );
		    
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
		}

		num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);

		load_soln_and_nghbr_coeffs<<<num_blocks_sol, THREADS_PER_BLOCK>>>
		(
			d_neighbours,
			d_scale_coeffs,
			d_assem_sol,
			solver_params
		);

		boundaries.update_all_inlets(input_filename, current_time);

		add_ghost_cells<<<num_blocks_sol, THREADS_PER_BLOCK>>>
		(
			d_assem_sol,
			d_neighbours,
			solver_params,
			sim_params,
			boundaries,
			current_time,
			dt,
			dx_finest,
			test_case
		);

		mra_end = clock();

		time_mra += (real)(mra_end - mra_start) / CLOCKS_PER_SEC;

		solver_start = clock();

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

			if (solver_params.limitslopes)
			{
				limit_slopes<<<num_blocks_sol, THREADS_PER_BLOCK>>>
				(
					d_assem_sol,
					d_neighbours,
					sim_params,
					solver_params,
					dx_finest,
					dy_finest,
					maxes.h
				);
			}

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

			reinsert_assem_sol<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_buf_assem_sol,
				d_assem_sol.act_idcs,
				d_scale_coeffs,
				solver_params
			);

			if ( solver_params.epsilon > C(0.0) )
			{
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
			}
			
			load_soln_and_nghbr_coeffs<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_neighbours,
				d_scale_coeffs,
				d_buf_assem_sol,
				solver_params
			);

			add_ghost_cells<<<num_blocks_sol, THREADS_PER_BLOCK>>>
			(
				d_buf_assem_sol,
				d_neighbours,
				solver_params,
				sim_params,
				boundaries,
				current_time,
				dt,
				dx_finest,
				test_case
			);

			rkdg2 = true;
			
			if (solver_params.limitslopes)
			{
				maxes.h = get_max_from_array
				(
					d_buf_assem_sol.h0, 
					d_buf_assem_sol.length
				);
				
				limit_slopes<<<num_blocks_sol, THREADS_PER_BLOCK>>>
				(
					d_buf_assem_sol,
					d_neighbours,
					sim_params,
					solver_params,
					dx_finest,
					dy_finest,
					maxes.h
				);
			}

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

		}

		dt = get_dt_CFL(d_dt_CFL, d_assem_sol.length);

		solver_end = clock();

		time_solver += (real)(solver_end - solver_start) / CLOCKS_PER_SEC;

		if ( saveint.save(current_time) )
		{
			if (plot_params.vtk)
			{
				write_soln_vtk
				(
					plot_params,
					d_assem_sol,
					d_dt_CFL,
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
					plot_params,
					d_assem_sol,
					dx_finest,
					dy_finest,
					sim_params,
					solver_params,
					saveint,
					first_timestep
				);
			}

			if (plot_params.c_prop)
			{
				write_c_prop_data
				(
					plot_params,
					start,
					solver_params,
					sim_params,
					d_assem_sol,
					current_time,
					time_mra,
					time_solver,
					dt,
					d_assem_sol.length,
					first_timestep
				);
			}
		}

		if ( massint.save(current_time) )
		{
			if (plot_params.cumulative)
			{
			    write_cumulative_data
			    (
			        start,
			        current_time,
			        time_mra,
			        time_solver,
					dt,
					d_assem_sol.length,
					sim_params,
			        plot_params,
			        first_timestep
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
			
			write_stage_point_data
			(
				mesh_dim,
				solver_params,
				plot_params,
				d_plot_assem_sol,
				p_finest_grid,
				stage_points,
				current_time,
				dx_finest,
				dy_finest,
				first_timestep
			);
		}

		if (timestep % 10000 == 1)
		{
			compression = C(100.0) - C(100.0) * d_assem_sol.length / (sim_params.xsz * sim_params.ysz);

			printf
			(
				"Elements: %d, compression: %f%%, time step: %f, timestep: %d, sim time: %f\n",
				d_assem_sol.length, compression, dt, timestep, current_time
			);
		}

		timestep++;
		
 		first_timestep = false;

		#if _RUN_UNIT_TESTS
		if (timestep > 2)
		{
			break;
		}
		#endif
	}

	end = clock();
	
	compression = C(100.0) - C(100.0) * d_assem_sol.length / (sim_params.xsz * sim_params.ysz);

	printf
	(
		"Elements: %d, compression: %f%%, time step: %f, timestep: %d, sim time: %f\n",
		d_assem_sol.length, compression, dt, timestep, current_time
	);
	
	run_time = (real)(end - start) / CLOCKS_PER_SEC;
	
	printf("Loop time: %f s\n", run_time);

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
}
