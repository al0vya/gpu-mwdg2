#include "compute_error_kernel.cuh"

__global__
void compute_error_kernel
(
	real* d_computed,
	real* d_verified,
	real* d_errors,
	int   array_length
)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < array_length)
	{
		d_errors[idx] = abs( d_verified[idx] - d_computed[idx] );
	}
}