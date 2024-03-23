#pragma once

#include <string>

#include "../classes/AssembledSolution.h"
#include "../classes/Neighbours.h"
#include "../classes/Details.h"
#include "../classes/ScaleCoefficients.h"
#include "../output/write_hierarchy_array_bool.cuh"

void generate_data_unit_test_preflag_topo
(
	const char*       dirroot,
	const char*       input_or_output_str,
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	SolverParams      solver_params
);

void generate_data_unit_test_encode_flow
(
	const char*       dirroot,
	const char*       input_or_output_str,
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	SolverParams      solver_params,
	const int&        timestep
);

void generate_data_unit_test_decoding
(
	const char*       dirroot,
	const char*       input_or_output_str,
	bool*             d_sig_details,
	real*             d_norm_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	const int&        timestep
);

void generate_data_unit_test_regularisation
(
	const char*  dirroot,
	const char*  input_or_output_str,
	bool*        d_sig_details,
	SolverParams solver_params,	
	const int&   timestep
);

void generate_data_unit_test_extra_significance
(
	const char*  dirroot,
	const char*  input_or_output_str,
	bool*        d_sig_details,
	real*        d_norm_details,
	SolverParams solver_params,	
	const int&   timestep
);

void generate_data_unit_test_dg2_update_RK1
(
	const char*        dirroot,
	const char*        input_or_output_str,
	Neighbours&        d_neighbours,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	const real&        dx_finest,
	const real&        dy_finest,
	const real&        dt,
	real*              d_dt_CFL,
	const int&         timestep
);