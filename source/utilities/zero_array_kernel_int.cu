#include "zero_array_kernel_real.cuh"

__global__
void zero_array_kernel_int
(
	int* d_array,
	int  num_threads
)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	d_array[idx] = 0;
}