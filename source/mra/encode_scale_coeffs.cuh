#pragma once

#include "cuda_runtime.h"

#include "../classes/ChildScaleCoeffs.h"
#include "../classes/ParentScaleCoeffs.h"
#include "encode_scale.cuh"

__device__ __forceinline__
ParentScaleCoeffsHW encode_scale_coeffs(const ChildScaleCoeffsHW& child_coeffs)
{
	return
	{
		encode_scale(child_coeffs.eta),
		encode_scale(child_coeffs.qx),
		encode_scale(child_coeffs.qy),
		encode_scale(child_coeffs.z)
	};
}

__device__ __forceinline__
ParentScaleCoeffsMW encode_scale_coeffs(const ChildScaleCoeffsMW& child_coeffs)
{
	return
	{
		{
			encode_scale_0(child_coeffs.eta),
			encode_scale_0(child_coeffs.qx),
			encode_scale_0(child_coeffs.qy),
			encode_scale_0(child_coeffs.z)
		},
		{
			encode_scale_1x(child_coeffs.eta),
			encode_scale_1x(child_coeffs.qx),
			encode_scale_1x(child_coeffs.qy),
			encode_scale_1x(child_coeffs.z)
		},
		{
			encode_scale_1y(child_coeffs.eta),
			encode_scale_1y(child_coeffs.qx),
			encode_scale_1y(child_coeffs.qy),
			encode_scale_1y(child_coeffs.z)
		}
	};
}