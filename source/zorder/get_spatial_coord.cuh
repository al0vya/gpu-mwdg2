#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__
real get_spatial_coord
(
	const int&  idx,
	const real& cellsize
);