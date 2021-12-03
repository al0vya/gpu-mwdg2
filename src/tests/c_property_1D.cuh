#pragma once

#include "cuda_runtime.h"

#include "real.h"

/*
 * 1D topography mimicking the c-property test case from GK's HFV1_MATLAB
 * see: https://github.com/ci1xgk/HFV1_MATLAB/tree/master/Test1_C_Pty
 * if x nodal values are input, topography will vary only along x-direction
 * likewise, if y nodal value is input, topography will vary only along y-direction
 */

__device__ __forceinline__ 
real c_property_1D(real x_or_y_int)
{
	real a = x_or_y_int;
	real z_int;

	if (a >= 22 && a < 25)
	{
		z_int = C(0.05) * a - C(1.1);
	}
	else if (a >= 25 && a <= 28)
	{
		z_int = C(-0.05) * a + C(1.4);
	}
	else if (a > 8 && a < 12)
	{
		z_int = C(0.2) - C(0.05) * (a - 10) * (a - 10);
	}
	else if (a > 39 && a < 46.5)
	{
		z_int = C(0.3);
	}
	else
	{
		z_int = 0;
	}
	
	return x_or_y_int > 20 ? 3 : 0;
}