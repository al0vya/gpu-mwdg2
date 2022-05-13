#pragma once

#include "cuda_runtime.h"

#include "ChildScaleCoefficients.h"
#include "ParentScaleCoefficient.h"
#include "encode_scale.cuh"

__device__
ParentScaleCoefficient encode_scale_coefficients(ChildScaleCoefficients child_scale_coeffs);