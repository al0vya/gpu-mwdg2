#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "Directions.h"

__device__
real get_y_face_coord
(
	const real& y,
	const real& dy_loc,
	const int&  direction
);