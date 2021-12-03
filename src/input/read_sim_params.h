#pragma once

#include <stdlib.h>
#include <iostream>

#include "SimulationParameters.h"
#include "SolverParameters.h"

SimulationParameters read_sim_params
(
	const int&              test_case,
	const char*             input_filename,
	const SolverParameters& solver_params
);