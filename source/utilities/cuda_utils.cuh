#pragma once

#include "cuda_runtime.h"

#include "CHECK_CUDA_ERROR.cuh"

__host__
cudaError_t sync();

__host__
cudaError_t peek();

__host__
cudaError_t reset();

__host__
cudaError_t copy
(
	void* dst,
	void* src,
	size_t bytes
);

__host__
cudaError_t copy_async
(
	void* dst,
	void* src,
	size_t bytes
);

__host__
void* malloc_device
(
	size_t bytes
);

__host__
cudaError_t free_device
(
	void* ptr
);

__host__
void* malloc_pinned
(
	size_t bytes
);

__host__
cudaError_t free_pinned
(
	void* ptr
);