#include "init_eta_temp.cuh"

__global__
void init_eta_temp
(
	AssembledSolution d_assem_sol,
	real*             d_eta_temp
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	d_eta_temp[idx] = d_assem_sol.h0[idx] + d_assem_sol.z0[idx];
}