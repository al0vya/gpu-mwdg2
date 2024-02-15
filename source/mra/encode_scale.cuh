#pragma once

#include "../types/real.h"
#include "../classes/ScaleChildren.h"
#include "../mra/Filters.h"

__host__ __device__ __forceinline__
real encode_scale(const ScaleChildrenHW& u)
{
	return C(0.5) * (H0 * (H0 * u.child_0 + H1 * u.child_2) + H1 * (H0 * u.child_1 + H1 * u.child_3));
}

__host__ __device__ __forceinline__
real encode_scale_0(const ScaleChildrenMW& u)
{
	return (HH0_11 * u._0.child_0 + HH0_12 * u._1x.child_0 + HH0_13 * u._1y.child_0 +
		    HH1_11 * u._0.child_2 + HH1_12 * u._1x.child_2 + HH1_13 * u._1y.child_2 +
		    HH2_11 * u._0.child_1 + HH2_12 * u._1x.child_1 + HH2_13 * u._1y.child_1 +
		    HH3_11 * u._0.child_3 + HH3_12 * u._1x.child_3 + HH3_13 * u._1y.child_3) / C(2.0);
}

__host__ __device__ __forceinline__
real encode_scale_1x(const ScaleChildrenMW& u)
{
	return (HH0_21 * u._0.child_0 + HH0_22 * u._1x.child_0 + HH0_23 * u._1y.child_0 +
		    HH1_21 * u._0.child_2 + HH1_22 * u._1x.child_2 + HH1_23 * u._1y.child_2 +
		    HH2_21 * u._0.child_1 + HH2_22 * u._1x.child_1 + HH2_23 * u._1y.child_1 +
		    HH3_21 * u._0.child_3 + HH3_22 * u._1x.child_3 + HH3_23 * u._1y.child_3) / C(2.0);
}

__host__ __device__ __forceinline__
real encode_scale_1y(const ScaleChildrenMW& u)
{
	return (HH0_31 * u._0.child_0 + HH0_32 * u._1x.child_0 + HH0_33 * u._1y.child_0 +
		    HH1_31 * u._0.child_2 + HH1_32 * u._1x.child_2 + HH1_33 * u._1y.child_2 +
		    HH2_31 * u._0.child_1 + HH2_32 * u._1x.child_1 + HH2_33 * u._1y.child_1 +
		    HH3_31 * u._0.child_3 + HH3_32 * u._1x.child_3 + HH3_33 * u._1y.child_3) / C(2.0);
}