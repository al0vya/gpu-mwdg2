#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__
void apply_friction
(
	const real& h,
	const real& tol_h,
	const real& tol_s,
	real&       qx,
	real&       qy,
	const real& n,
	const real& g,
	const real& dt
);