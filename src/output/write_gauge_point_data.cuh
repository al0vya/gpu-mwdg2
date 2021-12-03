#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "AssembledSolution.h"
#include "ScaleCoefficients.h"
#include "SolverParameters.h"
#include "compact.cuh"
#include "cuda_utils.cuh"
#include "GaugePoints.h"

#include "project_assem_sol.cuh"

__host__
void write_gauge_point_data
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParameters&  solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol,
	GaugePoints              gauge_points,
	const real&              time_now,
	const bool&              first_t_step
);