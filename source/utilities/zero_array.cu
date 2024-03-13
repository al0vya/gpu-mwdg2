#include "zero_array.cuh"

__global__
void zero_array
(
	real* d_array,
	int   num_threads
)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	d_array[idx] = C(0.0);
}