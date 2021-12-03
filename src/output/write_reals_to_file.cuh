#pragma once

#include "cuda_utils.cuh"

#include <stdio.h>
#include <string.h>

#include "real.h"

__host__
void write_reals_to_file
(
	const char* filename,
	const char* respath,
	real*       d_results,
	const int&  array_length
);