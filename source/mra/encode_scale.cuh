#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "ScaleChildren.h"
#include "Filters.h"

__device__ __forceinline__
real encode_scale(const ScaleChildrenHW& u)
{
	return C(0.5) * (H0 * (H0 * u.child_0 + H1 * u.child_2) + H1 * (H0 * u.child_1 + H1 * u.child_3));
}

__device__ __forceinline__
real encode_scale_0(const ScaleChildrenMW& u)
{
	return (HH0_11 * u._0.child_0 + HH0_12 * u._1x.child_0 + HH0_13 * u._1y.child_0 +
		    HH1_11 * u._0.child_2 + HH1_12 * u._1x.child_2 + HH1_13 * u._1y.child_2 +
		    HH2_11 * u._0.child_1 + HH2_12 * u._1x.child_1 + HH2_13 * u._1y.child_1 +
		    HH3_11 * u._0.child_3 + HH3_12 * u._1x.child_3 + HH3_13 * u._1y.child_3) / C(2.0);
}

__device__ __forceinline__
real encode_scale_1x(const ScaleChildrenMW& u)
{
	return (HH0_21 * u._0.child_0 + HH0_22 * u._1x.child_0 + HH0_23 * u._1y.child_0 +
		    HH1_21 * u._0.child_2 + HH1_22 * u._1x.child_2 + HH1_23 * u._1y.child_2 +
		    HH2_21 * u._0.child_1 + HH2_22 * u._1x.child_1 + HH2_23 * u._1y.child_1 +
		    HH3_21 * u._0.child_3 + HH3_22 * u._1x.child_3 + HH3_23 * u._1y.child_3) / C(2.0);
}

/*

GC0_31 * u0.child_0 + GC0_32 * u1x.child_0 + GC0_33 * u1y.child_0 +
GC1_31 * u0.child_2 + GC1_32 * u1x.child_2 + GC1_33 * u1y.child_2 +
GC2_31 * u0.child_1 + GC2_32 * u1x.child_1 + GC2_33 * u1y.child_1 +
GC3_31 * u0.child_3 + GC3_32 * u1x.child_3 + GC3_33 * u1y.child_3

GC0_31 * u._0.child_0 + GC0_32 * u._1x.child_0 + GC0_33 * u._1y.child_0
GC1_31 * u._0.child_2 + GC1_32 * u._1x.child_2 + GC1_33 * u._1y.child_2
GC2_31 * u._0.child_1 + GC2_32 * u._1x.child_1 + GC2_33 * u._1y.child_1
GC3_31 * u._0.child_3 + GC3_32 * u._1x.child_3 + GC3_33 * u._1y.child_3

*/

__device__ __forceinline__
real encode_scale_1y(const ScaleChildrenMW& u)
{
	return (HH0_31 * u._0.child_0 + HH0_32 * u._1x.child_0 + HH0_33 * u._1y.child_0 +
		    HH1_31 * u._0.child_2 + HH1_32 * u._1x.child_2 + HH1_33 * u._1y.child_2 +
		    HH2_31 * u._0.child_1 + HH2_32 * u._1x.child_1 + HH2_33 * u._1y.child_1 +
		    HH3_31 * u._0.child_3 + HH3_32 * u._1x.child_3 + HH3_33 * u._1y.child_3) / C(2.0);
}