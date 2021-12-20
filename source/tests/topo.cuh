#pragma once

#include "topo_c_prop_1D.cuh"
#include "topo_three_cones.cuh"
#include "topo_diff.cuh"
#include "topo_non_diff.cuh"
#include "topo_triangle.cuh"
#include "topo_parabolic_bowl.cuh"

// generate theoretical topography based on x, y nodal values
__device__ __forceinline__
real topo
(
	const real& x_int, 
	const real& y_int, 
	const int&  test_case
)
{
	real z_int = C(0.0);

	switch (test_case)
	{
	    case 1:
	    case 3:
	    	// variant along x-direction
	    	z_int = topo_c_prop_1D(x_int);
	    	break;
	    case 2:
	    case 4:
	    	// variant along y-direction
	    	z_int = topo_c_prop_1D(y_int);
	    	break;
	    case 11: // wet building overtopping x dir
		case 13: // wet-dry building overtopping x dir
	    	z_int = topo_c_prop_1D(x_int);
	    	break;
	    case 12: // wet building overtopping y dir
		case 14: // wet-dry building overtopping y dir
	    	z_int = topo_c_prop_1D(y_int);
	    	break;
	    case 15:
	    	z_int = topo_triangle(x_int);
	    	break;
	    case 16:
	    	z_int = topo_triangle(y_int);
	    	break;
	    case 17:
	    	// parabolic bowl in x dir
	    	// parabolic eqn is y = ax^2
	    	z_int = topo_parabolic_bowl(x_int);
	    	break;
	    case 18:
	    	// parabolic bowl in y dir
	    	z_int = topo_parabolic_bowl(y_int);
	    	break;
	    case 19:
	    	z_int = topo_three_cones(x_int, y_int);
	    	break;
	    case 20:
	    	z_int = topo_diff(x_int, y_int);
	    	break;
	    case 21:
	    	z_int = topo_non_diff(x_int, y_int);
	    	break;
	    default:
	    	break;
	}

	return z_int;
}