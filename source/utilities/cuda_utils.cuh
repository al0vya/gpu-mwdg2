#pragma once

#include "cuda_runtime.h"

#include "CHECK_CUDA_ERROR.cuh"

cudaError_t sync();

cudaError_t peek();

cudaError_t reset();

cudaError_t copy
(
	void* dst,
	void* src,
	size_t bytes
);

cudaError_t copy_async
(
	void* dst,
	void* src,
	size_t bytes
);

__host__ __device__
void* malloc_device
(
	size_t bytes
);

__host__ __device__
cudaError_t free_device
(
	void* ptr
);


__host__ __device__
void* malloc_pinned
(
	size_t bytes
);

__host__ __device__
cudaError_t free_pinned
(
	void* ptr
);