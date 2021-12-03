#include "get_compaction_flags.cuh"

__global__
void get_compaction_flags
(
	AssembledSolution d_assem_sol,
	CompactionFlags   d_compaction_flags,
	int               num_finest_elems
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_finest_elems) return;

	HierarchyIndex current = d_assem_sol.act_idcs[idx];

	if ( idx < (num_finest_elems - 1) )
	{
		HierarchyIndex right = d_assem_sol.act_idcs[idx + 1];

		d_compaction_flags.north_east[idx] = !(current == right);
	}
	else
	{
		d_compaction_flags.north_east[idx] = 1;
	}

	if (idx > 0)
	{
		HierarchyIndex left = d_assem_sol.act_idcs[idx - 1];

		d_compaction_flags.south_west[idx] = !(current == left);
	}
	else
	{
		d_compaction_flags.south_west[idx] = 1;
	}
}