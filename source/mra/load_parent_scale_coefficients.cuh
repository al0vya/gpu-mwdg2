#pragma once

#include "cuda_runtime.h"

#include "../classes/ScaleCoefficients.h"
#include "../classes/ParentScaleCoeffs.h"

__device__ __forceinline__ ParentScaleCoeffsHW load_parent_scale_coefficients
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
)
{
	ParentScaleCoeffsHW parent_coeffs =
	{
		d_scale_coeffs.eta0[h_idx],
		d_scale_coeffs.qx0[h_idx],
		d_scale_coeffs.qy0[h_idx],
		d_scale_coeffs.z0[h_idx]
	};

	return parent_coeffs;
}