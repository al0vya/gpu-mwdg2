#pragma once

#include "MortonCode.h"
#include "AssembledSolution.h"
#include "ScaleCoefficients.h"
#include "SolverParams.h"
#include "SaveInterval.h"
#include "FinestGrid.h"

#include "write_reals_to_file.cuh"

__host__
void write_soln_row_major
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParams&      solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol,
	SaveInterval&            saveint
);