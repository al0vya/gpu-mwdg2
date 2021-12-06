cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

include(CheckLanguage)

check_language(CUDA)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

project(proj-gpu-mwdg2)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gpu-mwdg2)

if (CMAKE_CUDA_COMPILER)
	
	enable_language(CUDA)
	
	set_property(TARGET gpu-mwdg2 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
	
	add_compile_definitions(CUDA THRUST_IGNORE_CUB_VERSION_CHECK)
	
	target_sources(
	    gpu-mwdg2
		PRIVATE
		src/MAIN.cu
		src/input/read_and_project_modes_dg2.cu
		src/input/read_and_project_modes_fv1.cu
		src/input/read_bound_conds.cpp
		src/input/read_cell_size.cpp
		src/input/read_raster_file.cpp
		src/input/read_respath.cpp
		src/input/read_save_interval.cpp
		src/input/read_sim_params.cpp
		src/input/read_solver_params.cpp
		src/input/read_test_case.cpp
		src/mra/decoding.cu
		src/mra/decoding_all.cu
		src/mra/encode_and_thresh_flow.cu
		src/mra/encode_and_thresh_topo.cu
		src/mra/encoding_all.cu
		src/mra/extra_significance.cu
		src/mra/get_max_scale_coeffs.cu
		src/mra/get_modal_values.cu
		src/mra/get_nodal_values.cu
		src/mra/get_reg_tree.cu
		src/mra/init_eta_temp.cu
		src/mra/init_nodal_values.cu
		src/mra/init_sig_details.cu
		src/mra/insert_point_srcs.cu
		src/mra/modal_projections.cu
		src/mra/preflag_details.cu
		src/mra/preflag_topo.cu
		src/mra/regularisation.cu
		src/mra/reinsert_assem_sol.cu
		src/mra/reinsert_point_srcs.cu
		src/mra/traverse_tree_of_sig_details.cu
		src/mra/zero_details.cu
		src/neighbours/add_ghost_cells.cu
		src/neighbours/compaction.cu
		src/neighbours/find_neighbours.cu
		src/neighbours/get_compaction_flags.cu
		src/neighbours/load_soln_and_nghbr_coeffs.cu
		src/operators/copy_to_buf_assem_sol.cu
		src/operators/dg2_update.cu
		src/operators/friction_implicit.cu
		src/operators/fv1_update.cu
		src/operators/get_dt_CFL.cu
		src/output/append_reals_to_file.cu
		src/output/project_assem_sol.cu
		src/output/write_all_raster_maps.cu
		src/output/write_bool_to_file.cu
		src/output/write_c_prop_data.cu
		src/output/write_cumu_sim_time.cpp
		src/output/write_gauge_point_data.cu
		src/output/write_int_to_file.cu
		src/output/write_mesh_info.cpp
		src/output/write_raster_file.cu
		src/output/write_reals_to_file.cu
		src/output/write_soln_row_major.cu
		src/output/write_soln_vtk.cu
		src/utilities/cuda_utils.cu
		src/utilities/get_num_blocks.cpp
		src/zorder/copy_finest_coefficients.cu
		src/zorder/generate_all_morton_codes.cu
		src/zorder/rev_z_order_act_idcs.cu
		src/zorder/rev_z_order_assem_sol.cu
		src/zorder/rev_z_order_reals.cu
		src/zorder/sort_finest_scale_coeffs_z_order.cu
		src/zorder/sort_neighbours_z_order.cu
	 "src/types (2)/Coordinate.h" "src/types (2)/Directions.h" "src/types (2)/HierarchyIndex.h" "src/types (2)/InletTypes.h" "src/types (2)/MortonCode.h" "src/types (2)/real.h" "src/types (2)/SolverTypes.h")
	
endif()

 target_include_directories(
     gpu-mwdg2
	 PRIVATE
	 #${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	 "src/classes/"
	 "src/input/"
	 "src/mra/"
	 "src/neighbours/"
	 "src/operators/"
	 "src/output/"
	 "src/tests/"
	 "src/types/"
	 "src/utilities/"
	 "src/zorder/"
	 )

if (CMAKE_BUILD_TYPE MATCHES Debug)
	target_compile_definitions(gpu-mwdg2 INTERFACE _DEBUG)
endif()