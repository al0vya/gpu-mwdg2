#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "Directions.h"

__device__
real get_x_face_coord
(
	const real& x,
	const real& dx_loc,
	const int&  direction
);