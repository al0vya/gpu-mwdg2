#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SimulationParameters.h"

void write_mesh_info
(
	const SimulationParameters& sim_params,
	const int&                  mesh_dim,
	const char*                 resdir
);