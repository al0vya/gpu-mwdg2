#pragma once

#include "read_keyword_int.h"
#include "read_keyword_real.h"
#include "read_keyword_bool.h"

#include "../classes/SolverParams.h"

SolverParams read_solver_params
(
	const char* input_filename
);