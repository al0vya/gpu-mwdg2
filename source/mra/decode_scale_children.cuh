#pragma once

#include "cuda_runtime.h"

#include "../mra/Filters.h"
#include "../classes/SubDetail.h"
#include "../classes/ScaleChildren.h"
#include "../classes/PlanarCoefficients.h"

__host__ __device__ __forceinline__
real decode_0(real& u, SubDetailHW& sub_detail);

__host__ __device__ __forceinline__
real decode_1(real& u, SubDetailHW& sub_detail);

__host__ __device__ __forceinline__
real decode_2(real& u, SubDetailHW& sub_detail);

__host__ __device__ __forceinline__
real decode_3(real& u, SubDetailHW& sub_detail);

__host__ __device__ __forceinline__
ScaleChildrenHW decode_scale_children
(
	real      u,
	SubDetailHW sub_detail
)
{
	return
	{
		decode_0(u, sub_detail), // child 0
		decode_1(u, sub_detail), // child 1
		decode_2(u, sub_detail), // child 2
		decode_3(u, sub_detail)  // child 3
	};
}

__host__ __device__ __forceinline__
real decode_0(real& u, SubDetailHW& sub_detail)
{
	return ( H0 * (H0 * u + G0 * sub_detail.alpha) + G0 * (H0 * sub_detail.beta + G0 * sub_detail.gamma) ) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2(real& u, SubDetailHW& sub_detail)
{
	return ( H0 * (H1 * u + G1 * sub_detail.alpha) + G0 * (H1 * sub_detail.beta + G1 * sub_detail.gamma) ) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1(real& u, SubDetailHW& sub_detail)
{
	return ( H1 * (H0 * u + G0 * sub_detail.alpha) + G1 * (H0 * sub_detail.beta + G0 * sub_detail.gamma) ) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3(real& u, SubDetailHW& sub_detail)
{
	return ( H1 * (H1 * u + G1 * sub_detail.alpha) + G1 * (H1 * sub_detail.beta + G1 * sub_detail.gamma) ) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH0_11 * u0  + HH0_21 * u1x  + HH0_31 * u1y  +
		   GA0_11 * da0 + GA0_21 * da1x + GA0_31 * da1y +
		   GB0_11 * db0 + GB0_21 * db1x + GB0_31 * db1y +
		   GC0_11 * dg0 + GC0_21 * dg1x + GC0_31 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH0_12 * u0  + HH0_22 * u1x  + HH0_32 * u1y  +
		   GA0_12 * da0 + GA0_22 * da1x + GA0_32 * da1y +
		   GB0_12 * db0 + GB0_22 * db1x + GB0_32 * db1y +
		   GC0_12 * dg0 + GC0_22 * dg1x + GC0_32 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH0_13 * u0  + HH0_23 * u1x  + HH0_33 * u1y  +
		   GA0_13 * da0 + GA0_23 * da1x + GA0_33 * da1y +
		   GB0_13 * db0 + GB0_23 * db1x + GB0_33 * db1y +
		   GC0_13 * dg0 + GC0_23 * dg1x + GC0_33 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH1_11 * u0  + HH1_21 * u1x  + HH1_31 * u1y  +
		   GA1_11 * da0 + GA1_21 * da1x + GA1_31 * da1y +
		   GB1_11 * db0 + GB1_21 * db1x + GB1_31 * db1y +
		   GC1_11 * dg0 + GC1_21 * dg1x + GC1_31 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH1_12 * u0  + HH1_22 * u1x  + HH1_32 * u1y  +
		   GA1_12 * da0 + GA1_22 * da1x + GA1_32 * da1y +
		   GB1_12 * db0 + GB1_22 * db1x + GB1_32 * db1y +
		   GC1_12 * dg0 + GC1_22 * dg1x + GC1_32 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_1y
(
	const real& u0,	 const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH1_13 * u0  + HH1_23 * u1x  + HH1_33 * u1y  +
		   GA1_13 * da0 + GA1_23 * da1x + GA1_33 * da1y +
		   GB1_13 * db0 + GB1_23 * db1x + GB1_33 * db1y +
		   GC1_13 * dg0 + GC1_23 * dg1x + GC1_33 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH2_11 * u0  + HH2_21 * u1x  + HH2_31 * u1y  +
		   GA2_11 * da0 + GA2_21 * da1x + GA2_31 * da1y +
		   GB2_11 * db0 + GB2_21 * db1x + GB2_31 * db1y +
		   GC2_11 * dg0 + GC2_21 * dg1x + GC2_31 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,	
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH2_12 * u0  + HH2_22 * u1x  + HH2_32 * u1y  +
		   GA2_12 * da0 + GA2_22 * da1x + GA2_32 * da1y +
		   GB2_12 * db0 + GB2_22 * db1x + GB2_32 * db1y +
		   GC2_12 * dg0 + GC2_22 * dg1x + GC2_32 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH2_13 * u0  + HH2_23 * u1x  + HH2_33 * u1y  +
		   GA2_13 * da0 + GA2_23 * da1x + GA2_33 * da1y +
		   GB2_13 * db0 + GB2_23 * db1x + GB2_33 * db1y +
		   GC2_13 * dg0 + GC2_23 * dg1x + GC2_33 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH3_11 * u0  + HH3_21 * u1x  + HH3_31 * u1y  +
		   GA3_11 * da0 + GA3_21 * da1x + GA3_31 * da1y +
		   GB3_11 * db0 + GB3_21 * db1x + GB3_31 * db1y +
		   GC3_11 * dg0 + GC3_21 * dg1x + GC3_31 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH3_12 * u0  + HH3_22 * u1x  + HH3_32 * u1y  +
		   GA3_12 * da0 + GA3_22 * da1x + GA3_32 * da1y +
		   GB3_12 * db0 + GB3_22 * db1x + GB3_32 * db1y +
		   GC3_12 * dg0 + GC3_22 * dg1x + GC3_32 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
)
{
	return (HH3_13 * u0  + HH3_23 * u1x  + HH3_33 * u1y  +
		   GA3_13 * da0 + GA3_23 * da1x + GA3_33 * da1y +
		   GB3_13 * db0 + GB3_23 * db1x + GB3_33 * db1y +
		   GC3_13 * dg0 + GC3_23 * dg1x + GC3_33 * dg1y) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_0
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH0_11 * s._0       + HH0_21 * s._1x       + HH0_31 * s._1y +
		    GA0_11 * d._0.alpha + GA0_21 * d._1x.alpha + GA0_31 * d._1y.alpha +
		    GB0_11 * d._0.beta  + GB0_21 * d._1x.beta  + GB0_31 * d._1y.beta +
		    GC0_11 * d._0.gamma + GC0_21 * d._1x.gamma + GC0_31 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_1x
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH0_12 * s._0       + HH0_22 * s._1x       + HH0_32 * s._1y +
		    GA0_12 * d._0.alpha + GA0_22 * d._1x.alpha + GA0_32 * d._1y.alpha +
		    GB0_12 * d._0.beta  + GB0_22 * d._1x.beta  + GB0_32 * d._1y.beta +
		    GC0_12 * d._0.gamma + GC0_22 * d._1x.gamma + GC0_32 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_0_1y
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH0_13 * s._0       + HH0_23 * s._1x       + HH0_33 * s._1y +
		    GA0_13 * d._0.alpha + GA0_23 * d._1x.alpha + GA0_33 * d._1y.alpha +
		    GB0_13 * d._0.beta  + GB0_23 * d._1x.beta  + GB0_33 * d._1y.beta +
		    GC0_13 * d._0.gamma + GC0_23 * d._1x.gamma + GC0_33 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_0
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH2_11 * s._0       + HH2_21 * s._1x       + HH2_31 * s._1y +
		    GA2_11 * d._0.alpha + GA2_21 * d._1x.alpha + GA2_31 * d._1y.alpha +
		    GB2_11 * d._0.beta  + GB2_21 * d._1x.beta  + GB2_31 * d._1y.beta +
		    GC2_11 * d._0.gamma + GC2_21 * d._1x.gamma + GC2_31 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_1x
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH2_12 * s._0       + HH2_22 * s._1x       + HH2_32 * s._1y +
		    GA2_12 * d._0.alpha + GA2_22 * d._1x.alpha + GA2_32 * d._1y.alpha +
		    GB2_12 * d._0.beta  + GB2_22 * d._1x.beta  + GB2_32 * d._1y.beta +
		    GC2_12 * d._0.gamma + GC2_22 * d._1x.gamma + GC2_32 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_1_1y
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH2_13 * s._0       + HH2_23 * s._1x       + HH2_33 * s._1y +
		    GA2_13 * d._0.alpha + GA2_23 * d._1x.alpha + GA2_33 * d._1y.alpha +
		    GB2_13 * d._0.beta  + GB2_23 * d._1x.beta  + GB2_33 * d._1y.beta +
		    GC2_13 * d._0.gamma + GC2_23 * d._1x.gamma + GC2_33 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_0
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH1_11 * s._0       + HH1_21 * s._1x       + HH1_31 * s._1y +
		    GA1_11 * d._0.alpha + GA1_21 * d._1x.alpha + GA1_31 * d._1y.alpha +
		    GB1_11 * d._0.beta  + GB1_21 * d._1x.beta  + GB1_31 * d._1y.beta +
		    GC1_11 * d._0.gamma + GC1_21 * d._1x.gamma + GC1_31 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_1x
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH1_12 * s._0       + HH1_22 * s._1x       + HH1_32 * s._1y +
		    GA1_12 * d._0.alpha + GA1_22 * d._1x.alpha + GA1_32 * d._1y.alpha +
		    GB1_12 * d._0.beta  + GB1_22 * d._1x.beta  + GB1_32 * d._1y.beta +
		    GC1_12 * d._0.gamma + GC1_22 * d._1x.gamma + GC1_32 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_2_1y
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH1_13 * s._0       + HH1_23 * s._1x       + HH1_33 * s._1y +
		    GA1_13 * d._0.alpha + GA1_23 * d._1x.alpha + GA1_33 * d._1y.alpha +
		    GB1_13 * d._0.beta  + GB1_23 * d._1x.beta  + GB1_33 * d._1y.beta +
		    GC1_13 * d._0.gamma + GC1_23 * d._1x.gamma + GC1_33 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_0
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH3_11 * s._0       + HH3_21 * s._1x       + HH3_31 * s._1y +
		    GA3_11 * d._0.alpha + GA3_21 * d._1x.alpha + GA3_31 * d._1y.alpha +
		    GB3_11 * d._0.beta  + GB3_21 * d._1x.beta  + GB3_31 * d._1y.beta +
		    GC3_11 * d._0.gamma + GC3_21 * d._1x.gamma + GC3_31 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_1x
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH3_12 * s._0       + HH3_22 * s._1x       + HH3_32 * s._1y +
		    GA3_12 * d._0.alpha + GA3_22 * d._1x.alpha + GA3_32 * d._1y.alpha +
		    GB3_12 * d._0.beta  + GB3_22 * d._1x.beta  + GB3_32 * d._1y.beta +
		    GC3_12 * d._0.gamma + GC3_22 * d._1x.gamma + GC3_32 * d._1y.gamma) * C(2.0);
}

__host__ __device__ __forceinline__
real decode_3_1y
(
	const PlanarCoefficients& s,
	const SubDetailMW&        d	
)
{
	return (HH3_13 * s._0       + HH3_23 * s._1x       + HH3_33 * s._1y +
		    GA3_13 * d._0.alpha + GA3_23 * d._1x.alpha + GA3_33 * d._1y.alpha +
		    GB3_13 * d._0.beta  + GB3_23 * d._1x.beta  + GB3_33 * d._1y.beta +
		    GC3_13 * d._0.gamma + GC3_23 * d._1x.gamma + GC3_33 * d._1y.gamma) * C(2.0);
}