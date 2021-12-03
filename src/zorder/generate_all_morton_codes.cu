#include "generate_all_morton_codes.cuh"

__global__
void generate_all_morton_codes
(
	MortonCode* d_morton_codes,
	int*        d_indices, 
	int         mesh_dim
)
{
	HierarchyIndex idx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( idx >= (mesh_dim * mesh_dim) ) return;

	Coordinate x = idx % mesh_dim;
	Coordinate y = idx / mesh_dim;

	d_indices[idx] = idx;

	d_morton_codes[idx] = generate_morton_code(x, y);
}