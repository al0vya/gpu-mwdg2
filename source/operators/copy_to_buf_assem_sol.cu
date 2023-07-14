#include "copy_to_buf_assem_sol.cuh"

__host__
void copy_to_buf_assem_sol
(
	const AssembledSolution& d_assem_sol,
	AssembledSolution&       d_buf_assem_sol
)
{
	size_t bytes_real     = d_assem_sol.length * sizeof(real);
	size_t bytes_int      = d_assem_sol.length * sizeof(int);
	size_t bytes_hier_idx = d_assem_sol.length * sizeof(HierarchyIndex);
	
	copy_cuda( d_buf_assem_sol.h0,        d_assem_sol.h0,        bytes_real );
	copy_cuda( d_buf_assem_sol.h1x,       d_assem_sol.h1x,       bytes_real );
	copy_cuda( d_buf_assem_sol.h1y,       d_assem_sol.h1y,       bytes_real );
	copy_cuda( d_buf_assem_sol.qx0,       d_assem_sol.qx0,       bytes_real );
	copy_cuda( d_buf_assem_sol.qx1x,      d_assem_sol.qx1x,      bytes_real );
	copy_cuda( d_buf_assem_sol.qx1y,      d_assem_sol.qx1y,      bytes_real );
	copy_cuda( d_buf_assem_sol.qy0,       d_assem_sol.qy0,       bytes_real );
	copy_cuda( d_buf_assem_sol.qy1x,      d_assem_sol.qy1x,      bytes_real );
	copy_cuda( d_buf_assem_sol.qy1y,      d_assem_sol.qy1y,      bytes_real );
	copy_cuda( d_buf_assem_sol.z0,        d_assem_sol.z0,        bytes_real );
	copy_cuda( d_buf_assem_sol.z1x,       d_assem_sol.z1x,       bytes_real );
	copy_cuda( d_buf_assem_sol.z1y,       d_assem_sol.z1y,       bytes_real );
	copy_cuda( d_buf_assem_sol.levels,    d_assem_sol.levels,    bytes_int );
	copy_cuda( d_buf_assem_sol.act_idcs,  d_assem_sol.act_idcs,  bytes_hier_idx );
			
	d_buf_assem_sol.length = d_assem_sol.length;
}