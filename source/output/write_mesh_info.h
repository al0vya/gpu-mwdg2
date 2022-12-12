#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../classes/SimulationParams.h"

void write_mesh_info
(
	const SimulationParams& sim_params,
	const int&                  mesh_dim,
	const char*                 resdir
);