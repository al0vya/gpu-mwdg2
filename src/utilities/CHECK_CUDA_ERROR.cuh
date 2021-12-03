#pragma once

#include "cuda_runtime.h"

#include <stdio.h>  // for fprintf
#include <stdlib.h> // for exit(error)

// macro to check for CUDA errors
#define CHECK_CUDA_ERROR(ans) { CUDAAssert( (ans), __FILE__, __LINE__); }

inline void CUDAAssert(cudaError_t error, const char* file, int line, bool abort = true)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s, %s, %d\n", cudaGetErrorString(error), file, line);

		if (abort) exit(error);
	}
}