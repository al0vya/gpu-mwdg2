#pragma once

#include <cstdlib>
#include <iostream>

#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"

#include "read_keyword_int.h"
#include "read_keyword_real.h"

SimulationParams read_sim_params
(
	const int&              test_case,
	const char*             input_filename,
	const SolverParams& solver_params
);