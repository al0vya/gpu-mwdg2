#pragma once

#include "cuda_runtime.h"

#include "LegendreBasis.h"

#include "get_x_face_coord.cuh"
#include "get_y_face_coord.cuh"
#include "get_x_face_unit.cuh"
#include "get_y_face_unit.cuh"

__device__
LegendreBasis get_leg_basis
(
	const HierarchyIndex& h_idx,
	const HierarchyIndex& h_idx_nghbr,
	const int&            level_nghbr,
	const int&            max_ref_lvl,
	const real&           x,
	const real&           y,
	const real&           dx_loc,
	const real&           dy_loc,
	const real&           dx_finest,
	const real&           dy_finest,
	const int&            direction
);