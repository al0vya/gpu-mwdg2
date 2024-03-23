#pragma once

#include "load_subchildren_warp.cuh"

__device__ __forceinline__
void load_children_warp
(
    ScaleChildrenMW&      children,
    real*                 d_s0,
    real*                 d_s1x,
    real*                 d_s1y,
    const HierarchyIndex& parent_idx,
    const HierarchyIndex& curr_lvl_idx,
    const HierarchyIndex& next_lvl_idx,
    const int&            lane_id
)
{
    load_subchildren_warp(children._0,  d_s0,  parent_idx, curr_lvl_idx, next_lvl_idx, lane_id);
    load_subchildren_warp(children._1x, d_s1x, parent_idx, curr_lvl_idx, next_lvl_idx, lane_id);
    load_subchildren_warp(children._1y, d_s1y, parent_idx, curr_lvl_idx, next_lvl_idx, lane_id);
}