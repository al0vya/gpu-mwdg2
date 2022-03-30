#include "cuda_utils.cuh"

__host__
cudaError_t sync()
{
	return cudaDeviceSynchronize();
}

__host__
cudaError_t peek()
{
	return cudaPeekAtLastError();
}

__host__
cudaError_t reset()
{
	return cudaDeviceReset();
}

__host__
cudaError_t copy
(
	void*  dst,
	void*  src,
	size_t bytes
)
{
	cudaError_t error = cudaMemcpy
	(
		dst,
		src,
		bytes,
		cudaMemcpyDefault
	);

	return error;
}

__host__
cudaError_t copy_async
(
	void*  dst,
	void*  src,
	size_t bytes
)
{
	cudaError_t error = cudaMemcpyAsync
	(
		dst,
		src,
		bytes,
		cudaMemcpyDefault
	);

	return error;
}

__host__
void* malloc_device
(
	size_t bytes
)
{
	void* ptr;
	
	cudaMalloc
	(
		&ptr,
		bytes
	);

	return ptr;
}

__host__
void* malloc_pinned
(
	size_t bytes
)
{
	void* ptr;

	cudaMallocHost
	(
		&ptr,
		bytes
	);

	return ptr;
}

__host__
cudaError_t free_device
(
	void* ptr
)
{
	return (nullptr != ptr) ? cudaFree(ptr) : cudaSuccess;
}

__host__
cudaError_t free_pinned
(
	void* ptr
)
{
	return (nullptr != ptr) ? cudaFreeHost(ptr) : cudaSuccess;
}