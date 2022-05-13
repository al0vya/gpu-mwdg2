#pragma once

#include "cuda_runtime.h"

#include "Filters.h"
#include "ParentScaleCoeffs.h"
#include "ChildScaleCoeffs.h"
#include "Detail.h"
#include "decode_scale_children.cuh"

__device__
ChildScaleCoeffsHW decode_scale_coefficients
(
	ParentScaleCoeffsHW& parent_coeffs,
	DetailHW&            detail
);