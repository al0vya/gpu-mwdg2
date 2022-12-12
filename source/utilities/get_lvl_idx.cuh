#pragma once

#include "cuda_runtime.h"

#include "../types/HierarchyIndex.h"

// gets the starting index of a grid at refinement level n within an array 
// containing grids at refinement level n to L, mapped to 1D using Morton codes,
// explained in comment: please grep 'ARRAY OF HIERARCHY OF GRIDS'
__host__ __device__ __forceinline__
HierarchyIndex get_lvl_idx(int level)
{
    /*
     * Prior explanation is in a comment: please grep 'ARRAY OF HIERARCHY OF GRIDS'.
     * The starting index in the array of grids can be found as follows.
     * A grid at refinement level n starts when the grid at refinement level n - 1 ends.
     * Up to the grid at n there are (4^(n+1) - 1) / 3 elements in the array.
     * Hence, up to a grid at n - 1 there are (4^n - 1) / 3 elements, which is the starting index.
     */

     // 1 << n = 2^n therefore 1 << 2 * n = 2^(2*n) = 4^n
    return ( ( 1 << (2 * level) ) - 1 ) / 3;
}