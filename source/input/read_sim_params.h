#pragma once

#include <cstdlib>
#include <iostream>

#include "SimulationParams.h"
#include "SolverParams.h"

#include "read_keyword_real.h"

SimulationParams read_sim_params
(
	const int&              test_case,
	const char*             input_filename,
	const SolverParams& solver_params
);