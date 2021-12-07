#pragma once

#include "cuda_runtime.h"

#include "ChildScaleCoefficients.h"
#include "ParentScaleCoefficient.h"
#include "encode_scale.cuh"

__device__ __forceinline__ ParentScaleCoefficient encode_scale_coefficients(ChildScaleCoefficients child_scale_coeffs)
{
	ParentScaleCoefficient parent_scale_coeffs =
	{
		encode_scale(child_scale_coeffs.eta),
		encode_scale(child_scale_coeffs.qx),
		encode_scale(child_scale_coeffs.qy),
		encode_scale(child_scale_coeffs.z)
	};

	return parent_scale_coeffs;
}