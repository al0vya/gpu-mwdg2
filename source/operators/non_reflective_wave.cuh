#pragma once

#include "cuda_runtime.h"

#include <cstdio>

#include "../types/real.h"

typedef struct twodoubles
{
	real hb, ub;

} twodoubles;

__device__ __forceinline__
twodoubles non_reflective_wave
(
	real h_inlet,
	real dt,
	real dxl,
	real hp,
	real up,
	real hb,
	real ub,
	real g
)
{
	int k = 0, max_k = 100, cond = 0;

	real Delta = C(1e-6), xa = -dxl / C(2.0), p0 = dxl / C(2.0), p1 = p0 - dxl / C(2.0), dp = C(0.0), relerr = C(0.0), f1 = C(0.0);

	real f0 = (p0 - xa) + dt * ((ub - (p0 - xa) / dxl * (ub - up)) - sqrt(g * (hb - (p0 - xa) / dxl * (hb - hp))));

	while ( (k <= max_k) && cond == 0 )
	{
		real df = C(1.0) + dt * ( (up - ub) / dxl - C(0.5) * pow((g * (hb - (p0 - xa) / dxl * (hb - hp))), C(-0.5)) * (g * (hp - hb) / dxl));

		if ( abs(df) <= C(0.0) )
		{
			printf("Convergence is doubtful because division by zero was encountered.\n");

			cond = 1;

			dp = p1 - p0;
			p1 = p0;
		}
		else
		{
			dp = f0 / df;
			p1 = p0 - dp;
		}

		f1     = (p1 - xa) + dt * ((ub - (p1 - xa) / dxl * (ub - up)) - sqrt(g * (hb - (p1 - xa) / dxl * (hb - hp))));
		relerr = abs(dp) / ( abs(p1) + C(0.0) );

		if (relerr < Delta)
		{
			cond = 2;

			//printf("The approximation p is within the desired tolerance.\n");
		}

		if (abs(f1) < Delta)
		{
			cond = 3;

			//printf("The computed function value fp is within the desired tolerance.\n");
		}

		if ( (relerr <= Delta) && (abs(f1) < Delta) )
		{
			cond = 4;

			//printf("The approximation p and the function value fp are both within the desired tolerance.\n");
		}

		p0 = p1;
		f0 = f1;
		k++;
	}

	real hr = hb - (hb - hp) * (p1 - xa) / dxl;
	real ur = ub - (ub - up) * (p1 - xa) / dxl;
	real cr = sqrt(g * hr);

	real beta_minus = ur - C(2.0) * cr;
	real beta_plus  = C(2.0) * sqrt(g * (h_inlet + C(0.13535))) + h_inlet * sqrt(g / C(0.13535));
	
	// this is "hb" in MKS Fortran code 
	real h_out = C(1.0) / C(16.0) / g * (beta_plus - beta_minus) * (beta_plus - beta_minus);
	real u_out = C(0.5) * (beta_plus + beta_minus);

	return { h_out, u_out };
}