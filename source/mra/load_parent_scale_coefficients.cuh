#pragma once

#include "cuda_runtime.h"

#include "ScaleCoefficients.h"
#include "ParentScaleCoeffs.h"

__device__
ParentScaleCoeffsHW load_parent_scale_coefficients
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
);