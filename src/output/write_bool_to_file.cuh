#pragma once

#include "cuda_utils.cuh"

#include <stdio.h>
#include <string.h>

__host__
void write_bool_to_file
(
	const char* filename,
	const char* respath,
	bool*       d_results,
	const int   array_length
);