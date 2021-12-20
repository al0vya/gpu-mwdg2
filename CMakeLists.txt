cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)
enable_language(CUDA)
	
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(proj-gpu-mwdg2)

add_executable(gpu-mwdg2)

if (CMAKE_CUDA_COMPILER)
	
	set_property(TARGET gpu-mwdg2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	
	add_compile_definitions(CUDA THRUST_IGNORE_CUB_VERSION_CHECK)
	
	target_sources(
	    gpu-mwdg2
		PRIVATE
		source/MAIN.cu
		source/input/read_and_project_modes_dg2.cu
		source/input/read_and_project_modes_fv1.cu
		source/input/read_bound_conds.cpp
		source/input/read_cell_size.cpp
		source/input/read_keyword_int.cpp
		source/input/read_keyword_real.cpp
		source/input/read_keyword_str.cpp
		source/input/read_plot_params.cpp
		source/input/read_raster_file.cpp
		source/input/read_respath.cpp
		source/input/read_save_interval.cpp
		source/input/read_sim_params.cpp
		source/input/read_solver_params.cpp
		source/input/read_test_case.cpp
		source/mra/decoding.cu
		source/mra/decoding_all.cu
		source/mra/encode_and_thresh_flow.cu
		source/mra/encode_and_thresh_topo.cu
		source/mra/encoding_all.cu
		source/mra/extra_significance.cu
		source/mra/get_max_scale_coeffs.cu
		source/mra/get_modal_values.cu
		source/mra/get_nodal_values.cu
		source/mra/get_reg_tree.cu
		source/mra/init_eta_temp.cu
		source/mra/init_nodal_values.cu
		source/mra/init_sig_details.cu
		source/mra/insert_point_srcs.cu
		source/mra/modal_projections.cu
		source/mra/preflag_details.cu
		source/mra/preflag_topo.cu
		source/mra/regularisation.cu
		source/mra/reinsert_assem_sol.cu
		source/mra/reinsert_point_srcs.cu
		source/mra/traverse_tree_of_sig_details.cu
		source/mra/zero_details.cu
		source/neighbours/add_ghost_cells.cu
		source/neighbours/compaction.cu
		source/neighbours/find_neighbours.cu
		source/neighbours/get_compaction_flags.cu
		source/neighbours/load_soln_and_nghbr_coeffs.cu
		source/operators/copy_to_buf_assem_sol.cu
		source/operators/dg2_update.cu
		source/operators/friction_implicit.cu
		source/operators/fv1_update.cu
		source/operators/get_dt_CFL.cu
		source/output/append_reals_to_file.cu
		source/output/project_assem_sol.cu
		source/output/write_all_raster_maps.cu
		source/output/write_bool_to_file.cu
		source/output/write_c_prop_data.cu
		source/output/write_cumu_sim_time.cpp
		source/output/write_gauge_point_data.cu
		source/output/write_int_to_file.cu
		source/output/write_mesh_info.cpp
		source/output/write_raster_file.cu
		source/output/write_reals_to_file.cu
		source/output/write_soln_planar.cu
		source/output/write_soln_planar_dg2.cu
		source/output/write_soln_planar_fv1.cu
		source/output/write_soln_row_major.cu
		source/output/write_soln_vtk.cu
		source/utilities/cuda_utils.cu
		source/utilities/get_num_blocks.cpp
		source/zorder/copy_finest_coefficients.cu
		source/zorder/generate_all_morton_codes.cu
		source/zorder/rev_z_order_act_idcs.cu
		source/zorder/rev_z_order_assem_sol.cu
		source/zorder/rev_z_order_reals.cu
		source/zorder/sort_finest_scale_coeffs_z_order.cu
		source/zorder/sort_neighbours_z_order.cu
	)
	
endif()

 target_include_directories(
     gpu-mwdg2
	 PRIVATE
	 #${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	 "source/classes/"
	 "source/input/"
	 "source/mra/"
	 "source/neighbours/"
	 "source/operators/"
	 "source/output/"
	 "source/tests/"
	 "source/types/"
	 "source/utilities/"
	 "source/zorder/"
	 )