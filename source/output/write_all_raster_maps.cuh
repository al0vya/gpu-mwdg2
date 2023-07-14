#pragma once

#include "write_all_raster_maps_fv1.cuh"
#include "write_all_raster_maps_dg2.cuh"

void write_all_raster_maps
(
    const PlottingParams&    plot_params,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint,
	const bool&              first_t_step
);