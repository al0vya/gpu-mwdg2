#pragma once

#include "cuda_runtime.h"

#include "ParentScaleCoeffs.h"
#include "ChildScaleCoeffs.h"
#include "ScaleCoefficients.h"
#include "HierarchyIndex.h"

__device__
void store_scale_coeffs
(
	const ParentScaleCoeffsHW& parent_coeffs, 
	const ScaleCoefficients&   d_scale_coeffs,
	const HierarchyIndex&      h_idx
);

__device__
void store_scale_coeffs
(
	const ParentScaleCoeffsMW& parent_coeffs,
	const ScaleCoefficients&   d_scale_coeffs,
	const HierarchyIndex&      h_idx
);

__device__
void store_scale_coeffs
(
	const ChildScaleCoeffsHW& child_coeffs, 
	const ScaleCoefficients&  d_scale_coeffs,
	const HierarchyIndex&     h_idx
);

__device__
void store_scale_coeffs
(
	const ChildScaleCoeffsMW& child_coeffs, 
	const ScaleCoefficients&  d_scale_coeffs,
	const HierarchyIndex&     h_idx
);