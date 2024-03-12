#pragma once

#include "cuda_runtime.h"

#include "../classes/ScaleCoefficients.h"
#include "../classes/ScaleChildren.h"

__device__ __forceinline__
void load_subchildren_warp
(
    ScaleChildrenHW&      subchildren,
    real*                 d_s0_1x_1y,
    const HierarchyIndex& parent_idx,
    const HierarchyIndex& curr_lvl_idx,
    const HierarchyIndex& next_lvl_idx,
    const int&            lane_id
)
{
    const unsigned int mask = 0xffffffff;

    real s[4];

    for (int starting_lane_id = 0; starting_lane_id < 32; starting_lane_id = starting_lane_id + 8)
    {
        HierarchyIndex child_idx = 4 * (__shfl_sync(mask, parent_idx, starting_lane_id + lane_id / 4) - curr_lvl_idx) + next_lvl_idx + lane_id % 4;

        real val = d_s0_1x_1y[child_idx];

        __syncwarp();

        s[0] = __shfl_sync(mask, val, 4 * (lane_id % 8) + 0);
        s[1] = __shfl_sync(mask, val, 4 * (lane_id % 8) + 1);
        s[2] = __shfl_sync(mask, val, 4 * (lane_id % 8) + 2);
        s[3] = __shfl_sync(mask, val, 4 * (lane_id % 8) + 3);

        if (lane_id >= starting_lane_id && lane_id < starting_lane_id + 8)
        {
            subchildren.child_0 = s[0];
            subchildren.child_1 = s[1];
            subchildren.child_2 = s[2];
            subchildren.child_3 = s[3];
        }

        __syncwarp();
    }
}