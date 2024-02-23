#pragma once

#include <string>

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

void generate_data_unit_test_encoding_all
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