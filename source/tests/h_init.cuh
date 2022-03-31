#pragma once

#include "h_init_overtopping.cuh"
#include "h_init_c_property.cuh"
#include "h_init_radial.cuh"
#include "h_init_three_peaks.cuh"
#include "h_init_three_blocks.cuh"
#include "h_init_parabolic_bowl.cuh"

__device__ __forceinline__
real h_init
(
	const real& x_int,
	const real& y_int,
	const real& z_int,
	const Depths1D& bcs,
	const int& test_case
)
{
	real h_int = 0;

	switch (test_case)
	{
	    case 1: // wet c prop x dir
	    case 3: // wet-dry c prop x dir
	    	h_int = h_init_c_property(bcs, z_int, x_int);
	    	break;
	    case 2: // wet c prop y dir
	    case 4: // wet-dry c prop y dir
	    	h_int = h_init_c_property(bcs, z_int, y_int);
	    	break;
	    case 5:  // wet dam break x dir
	    case 7:  // dry dam break x dir
	    case 9:  // dry dam break x dir w fric
	    case 11: // wet building overtopping x dir
	    case 13: // dry building overtopping x dir
	    	h_int = h_init_overtopping(bcs, z_int, x_int);
	    	break;
	    case 6:  // wet dam break y dir
	    case 8:  // dry dam break y dir
	    case 10: // dry dam break y dir w fric
	    case 12: // wet building overtopping y dir
	    case 14: // dry building overtopping y dir
	    	h_int = h_init_overtopping(bcs, z_int, y_int);
	    	break;
	    case 15: // triangular dam break x dir
	    	h_int = ( x_int < C(15.5) ) ? C(0.75) : C(0.0);
	    	break;
	    case 16: // triangular dam break y dir
	    	h_int = ( y_int < C(15.5) ) ? C(0.75) : C(0.0);
	    	break;
	    case 17: // parabolic bowl x dir
			h_int = h_init_parabolic_bowl(x_int, z_int);
	    	break;
	    case 18: // parabolic bowl y dir
	    	h_int = h_init_parabolic_bowl(y_int, z_int);
	    	break;
	    case 19: // three cones
	    	h_int = C(1.0) - z_int;
	    	break;
	    case 20: // three cones dam break
	    	h_int = ( x_int < C(16.0) ) ? max( (C(1.875) - z_int), C(0.0) ) : C(0.0);
	    	break;
	    case 21: // differential geometry
	    	h_int = C(1.95) - z_int;
	    	break;
	    case 22: // non-differential geometry
	    	h_int = C(1.78) - z_int;
	    	break;
	    case 23: // radial dam break
	    	h_int = h_init_radial(x_int, y_int);
	    	break;
	    default:
	    	break;
	}

	return h_int;
}