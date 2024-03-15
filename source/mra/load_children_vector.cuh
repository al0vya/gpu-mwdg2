#pragma once

#include "cuda_runtime.h"

#include "../classes/ScaleCoefficients.h"
#include "../classes/ScaleChildren.h"

__device__ __forceinline__
void load_children_vector
(
    ScaleChildrenHW&      children,
    real*                 d_s0,
    const HierarchyIndex& child_idx
)
{
    real4 s;
    
    s = *reinterpret_cast<real4*>(d_s0 + child_idx);

    children.child_0 = s.x;
    children.child_1 = s.y;
    children.child_2 = s.z;
    children.child_3 = s.w;
}

__device__ __forceinline__
void load_children_vector
(
    ScaleChildrenMW&      children,
    real*                 d_s0,
    real*                 d_s1x,
    real*                 d_s1y,
    const HierarchyIndex& child_idx
)
{
    real4 s;
    
    s = *reinterpret_cast<real4*>(d_s0 + child_idx);

    children._0.child_0 = s.x;
    children._0.child_1 = s.y;
    children._0.child_2 = s.z;
    children._0.child_3 = s.w;

    s = *reinterpret_cast<real4*>(d_s1x + child_idx);

    children._1x.child_0 = s.x;
    children._1x.child_1 = s.y;
    children._1x.child_2 = s.z;
    children._1x.child_3 = s.w;

    s = *reinterpret_cast<real4*>(d_s1y + child_idx);

    children._1y.child_0 = s.x;
    children._1y.child_1 = s.y;
    children._1y.child_2 = s.z;
    children._1y.child_3 = s.w;
}