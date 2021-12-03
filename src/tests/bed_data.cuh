#pragma once

#include "c_property_1D.cuh"
#include "three_cones.cuh"
#include "diff_topo.cuh"
#include "non_diff_topo.cuh"
#include "triangle_topo.cuh"

// generate theoretical topography based on x, y nodal values
__device__ __forceinline__
real bed_data
(
	const real& x_int, 
	const real& y_int, 
	const int&  test_case
)
{
	real z_int = 0;

	switch (test_case)
	{
	case 1:
		// variant along x-direction
		z_int = c_property_1D(x_int);
		break;
	case 2:
		// variant along y-direction
		z_int = c_property_1D(y_int);
		break;
	case 3:
		z_int = three_cones(x_int, y_int);
		break;
	case 10:
		z_int = c_property_1D(x_int);
		break;
	case 11:
		z_int = c_property_1D(y_int);
		break;
	case 13:
		z_int = diff_topo(x_int, y_int);
		break;
	case 14:
		z_int = non_diff_topo(x_int, y_int);
		break;
	case 15:
		z_int = triangle_topo(x_int);
		break;
	case 16:
		z_int = triangle_topo(y_int);
		break;
	case 17:
		// parabolic bowl in x dir
		// parabolic eqn is y = ax^2
		z_int = C(0.01) * x_int * x_int;
		break;
	case 18:
		// parabolic bowl in y dir
		z_int = C(0.01) * y_int * y_int;
		break;
	default:
		break;
	}

	return z_int;
}