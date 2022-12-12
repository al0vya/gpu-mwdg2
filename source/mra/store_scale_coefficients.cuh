#pragma once

#include "cuda_runtime.h"

#include <cstdio>

#include "../classes/ParentScaleCoeffs.h"
#include "../classes/ChildScaleCoeffs.h"
#include "../classes/ScaleCoefficients.h"
#include "index_1D.h"

__device__ __forceinline__ void store_scale_coefficients
(
	const ParentScaleCoeffsHW& parent_coeffs, 
	const ScaleCoefficients&   d_scale_coeffs,
	const index_1D&            h_idx
)
{
	d_scale_coeffs.eta0[h_idx] = parent_coeffs.eta;
	d_scale_coeffs.qx0[h_idx]  = parent_coeffs.qx;
	d_scale_coeffs.qy0[h_idx]  = parent_coeffs.qy;
}

__device__ __forceinline__ void store_scale_coefficients
(
	const ParentScaleCoeffsMW& parent_coeffs,
	const ScaleCoefficients&   d_scale_coeffs,
	const index_1D& h_idx
)
{
	d_scale_coeffs.eta0[h_idx] = parent_coeffs._0.eta;
	d_scale_coeffs.qx0[h_idx]  = parent_coeffs._0.qx;
	d_scale_coeffs.qy0[h_idx]  = parent_coeffs._0.qy;
	
	d_scale_coeffs.eta1x[h_idx] = parent_coeffs._1x.eta;
	d_scale_coeffs.qx1x[h_idx]  = parent_coeffs._1x.qx;
	d_scale_coeffs.qy1x[h_idx]  = parent_coeffs._1x.qy;

	d_scale_coeffs.eta1y[h_idx] = parent_coeffs._1y.eta;
	d_scale_coeffs.qx1y[h_idx]  = parent_coeffs._1y.qx;
	d_scale_coeffs.qy1y[h_idx]  = parent_coeffs._1y.qy;
}

__device__ __forceinline__ void store_scale_coefficients
(
	const ChildScaleCoeffsHW& child_coeffs, 
	const ScaleCoefficients&  d_scale_coeffs,
	const index_1D&           h_idx
)
{
	d_scale_coeffs.eta0[h_idx + 0] = child_coeffs.eta.child_0;
	d_scale_coeffs.eta0[h_idx + 1] = child_coeffs.eta.child_1;
	d_scale_coeffs.eta0[h_idx + 2] = child_coeffs.eta.child_2;
	d_scale_coeffs.eta0[h_idx + 3] = child_coeffs.eta.child_3;

	d_scale_coeffs.qx0[h_idx + 0]  = child_coeffs.qx.child_0;
	d_scale_coeffs.qx0[h_idx + 1]  = child_coeffs.qx.child_1;
	d_scale_coeffs.qx0[h_idx + 2]  = child_coeffs.qx.child_2;
	d_scale_coeffs.qx0[h_idx + 3]  = child_coeffs.qx.child_3;

	d_scale_coeffs.qy0[h_idx + 0]  = child_coeffs.qy.child_0;
	d_scale_coeffs.qy0[h_idx + 1]  = child_coeffs.qy.child_1;
	d_scale_coeffs.qy0[h_idx + 2]  = child_coeffs.qy.child_2;
	d_scale_coeffs.qy0[h_idx + 3]  = child_coeffs.qy.child_3;

	/*d_scale_coeffs.z0[h_idx + 0]   = child_coeffs.z.child_0;
	d_scale_coeffs.z0[h_idx + 1]   = child_coeffs.z.child_1;
	d_scale_coeffs.z0[h_idx + 2]   = child_coeffs.z.child_2;
	d_scale_coeffs.z0[h_idx + 3]   = child_coeffs.z.child_3;*/
}