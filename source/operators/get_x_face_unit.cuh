#pragma once

#include "cuda_runtime.h"

#include "Directions.h"
#include "real.h"
#include "HierarchyIndex.h"

#include "get_x_coord.cuh"

__device__
real get_x_face_unit
(
	const HierarchyIndex& h_idx,
	const HierarchyIndex& h_idx_nghbr,
	const int&            level_nghbr,
	const int&            max_ref_lvl,
	const real&           x_face, 
	const real&           dx_finest, 
	const int&            direction
);