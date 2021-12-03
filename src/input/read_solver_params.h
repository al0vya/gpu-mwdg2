#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "SolverParameters.h"

SolverParameters read_solver_params
(
	const char* input_filename
);