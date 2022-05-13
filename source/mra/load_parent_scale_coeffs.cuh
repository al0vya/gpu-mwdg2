#pragma once

#include "cuda_runtime.h"

#include "ScaleCoefficients.h"
#include "ParentScaleCoeffs.h"

__device__ __forceinline__
ParentScaleCoeffsHW load_parent_scale_coeffs_hw
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
)
{
	return
	{
		d_scale_coeffs.eta0[h_idx],
		d_scale_coeffs.qx0[h_idx],
		d_scale_coeffs.qy0[h_idx],
		d_scale_coeffs.z0[h_idx]
	};
}

__device__ __forceinline__ 
ParentScaleCoeffsMW load_parent_scale_coeffs_mw
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
)
{
	return
	{
		{
			d_scale_coeffs.eta0[h_idx],
			d_scale_coeffs.qx0[h_idx],
			d_scale_coeffs.qy0[h_idx],
			d_scale_coeffs.z0[h_idx]
		},
		{
			d_scale_coeffs.eta1x[h_idx],
			d_scale_coeffs.qx1x[h_idx],
			d_scale_coeffs.qy1x[h_idx],
			d_scale_coeffs.z1x[h_idx]
		},
		{
			d_scale_coeffs.eta1y[h_idx],
			d_scale_coeffs.qx1y[h_idx],
			d_scale_coeffs.qy1y[h_idx],
			d_scale_coeffs.z1y[h_idx]
		}
	};
}