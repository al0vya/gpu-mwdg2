#pragma once

// Kernels
#include "zorder/generate_all_morton_codes.cuh"
#include "zorder/copy_finest_coefficients.cuh"
#include "mra/insert_point_srcs.cuh"
#include "mra/reinsert_point_srcs.cuh"
#include "mra/init_sig_details.cuh"
#include "mra/zero_details.cuh"
#include "mra/traverse_tree_of_sig_details.cuh"
#include "neighbours/find_neighbours.cuh"
#include "neighbours/get_compaction_flags.cuh"
#include "neighbours/load_soln_and_nghbr_coeffs.cuh"
#include "neighbours/add_ghost_cells.cuh"
#include "operators/friction_implicit.cuh"
#include "operators/limit_slopes.cuh"
#include "operators/fv1_update.cuh"
#include "operators/dg2_update.cuh"

// Kernel wrappers
#include "mra/get_nodal_values.cuh"
#include "mra/get_modal_values.cuh"
#include "zorder/sort_finest_scale_coeffs_z_order.cuh"
#include "mra/get_max_scale_coeffs.cuh"
#include "mra/preflag_topo.cuh"
#include "mra/encode_flow.cuh"
#include "mra/get_reg_tree.cuh"
#include "mra/extra_significance.cuh"
#include "mra/decoding.cuh"
#include "zorder/rev_z_order_act_idcs.cuh"
#include "zorder/rev_z_order_reals.cuh"
#include "zorder/sort_neighbours_z_order.cuh"
#include "neighbours/compaction.cuh"
#include "operators/get_dt_CFL.cuh"

// Input/output
#include "input/read_cell_size.h"
#include "input/read_command_line_params.h"
#include "input/read_test_case.h"
#include "output/write_all_raster_maps.cuh"
#include "output/write_c_prop_data.cuh"
#include "output/write_mesh_info.h"
#include "output/write_stage_point_data.cuh"
#include "output/write_soln_vtk.cuh"
#include "output/write_hierarchy_array_bool.cuh"

// Helper functions
#include "utilities/get_lvl_idx.cuh"
#include "mra/preflag_details.cuh"
#include "output/project_assem_sol.cuh"
#include "operators/copy_to_buf_assem_sol.cuh"

// Sorting
#include "zorder/get_sorting_indices.cuh"

// Unit tests
#include "unittests/generate_data_unit_tests.cuh"
#include "unittests/run_unit_tests.cuh"

void run_simulation
(
	int    argc,
	char** argv
);