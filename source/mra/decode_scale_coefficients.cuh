#pragma once

#include "cuda_runtime.h"

#include "Filters.h"
#include "ParentScaleCoeffs.h"
#include "ChildScaleCoeffs.h"
#include "Detail.h"
#include "decode_scale_children.cuh"

__device__ __forceinline__ ChildScaleCoeffsHW decode_scale_coefficients
(
	ParentScaleCoeffsHW& parent_coeffs,
	DetailHW&            detail
)
{
	return
	{
		decode_scale_children(parent_coeffs.eta, detail.eta),
		decode_scale_children(parent_coeffs.qx,  detail.qx),
		decode_scale_children(parent_coeffs.qy,  detail.qy),
		{ 0, 0, 0, 0 }
	};
}