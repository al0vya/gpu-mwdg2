#include "get_lvl_idx.cuh"

__host__ __device__
HierarchyIndex get_lvl_idx(int level)
{
    return ( ( 1 << (2 * level) ) - 1 ) / 3;
}