#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Depths1D.h"
#include "SimulationParams.h"
#include "NodalValues.h"
#include "HierarchyIndex.h"
#include "Coordinate.h"

#include "h_init.cuh"
#include "bed_data.cuh"

// initialise nodal values of h, qx, qy and z depending on x, y nodal values
__global__
void init_nodal_values
(
	NodalValues          d_nodal_vals,
	real                 dx_finest,
	real                 dy_finest,
	Depths1D             bcs,
	SimulationParams sim_params,
	int                  interface_dim,
	int                  test_case
);