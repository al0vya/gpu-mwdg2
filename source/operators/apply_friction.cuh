#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"

__device__ __forceinline__
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
)
{
	bool below_depth = (h < tol_h);
	
	if (below_depth)
	{
		qx = C(0.0);
		qy = C(0.0);

		return;
	}
	
	real ux = qx / h;
	real uy = qy / h;

	if (abs(ux) < tol_s && abs(uy) < tol_s)
	{
		qx = C(0.0);
		qy = C(0.0);

		return;
	}

	real Cf = g * n * n / pow( h, C(1.0) / C(3.0) );

	real speed = sqrt(ux * ux + uy * uy);

	real Sf_x = -Cf * ux * speed;
	real Sf_y = -Cf * uy * speed;

	real D_x = C(1.0) + dt * Cf / h * (C(2.0) * ux * ux + uy * uy) / speed;
	real D_y = C(1.0) + dt * Cf / h * (C(2.0) * uy * uy + ux * ux) / speed;

	qx += dt * Sf_x / D_x;
	qy += dt * Sf_y / D_y;
}