#include "init_sig_details.cuh"

__global__
void init_sig_details
(
	bool* d_sig_details,
	int   num_details
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_details) return;

	d_sig_details[idx] = true;
}