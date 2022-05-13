#pragma once

#include "cuda_runtime.h"

#include "ScaleCoefficients.h"
#include "ParentScaleCoeffs.h"

__device__
ParentScaleCoeffsHW load_parent_scale_coeffs_hw
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
);

__device__
ParentScaleCoeffsMW load_parent_scale_coeffs_mw
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
);