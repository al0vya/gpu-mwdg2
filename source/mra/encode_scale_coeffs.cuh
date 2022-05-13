#pragma once

#include "cuda_runtime.h"

#include "ChildScaleCoeffs.h"
#include "ParentScaleCoeffs.h"
#include "encode_scale.cuh"

__device__
ParentScaleCoeffsHW encode_scale_coeffs(const ChildScaleCoeffsHW& child_coeffs);

__device__
ParentScaleCoeffsMW encode_scale_coeffs(const ChildScaleCoeffsMW& child_coeffs);