#pragma once

#include "cuda_runtime.h"

#include <cstdio>

#include "real.h"

typedef struct twodoubles
{
	real hb, ub;

} twodoubles;

__device__
twodoubles non_reflective_wave
(
	const real& h_inlet,
	const real& dt,
	const real& dxl,
	const real& hp,
	const real& up,
	const real& hb,
	const real& ub,
	const real& g
);