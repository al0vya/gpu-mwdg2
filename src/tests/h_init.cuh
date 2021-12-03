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
	const real&     x_int,
	const real&     y_int,
	const real&     z_int,
	const Depths1D& bcs,
	const int&      test_case
)
{
	real h_int = 0;

	switch (test_case)
	{
	case 1: // c prop x dir
		h_int = h_init_c_property(bcs, z_int, x_int);
		break;
	case 2: // c prop y dir
		h_int = h_init_c_property(bcs, z_int, y_int);
		break;
	case 3: // three cones
		h_int = C(0.875) - z_int;
		break;
	case 4: // wet dam break x dir
		h_int = h_init_overtopping(bcs, z_int, x_int);
		break;
	case 5: // wet dam break y dir
		h_int = h_init_overtopping(bcs, z_int, y_int);
		break;
	case 6: // dry dam break x dir
		h_int = h_init_overtopping(bcs, z_int, x_int);
		break;
	case 7: // dry dam break y dir
		h_int = h_init_overtopping(bcs, z_int, y_int);
		break;
	case 8: // dry dam break fric x dir
		h_int = h_init_overtopping(bcs, z_int, x_int);
		break;
	case 9: // dry dam break fric y dir
		h_int = h_init_overtopping(bcs, z_int, y_int);
		break;
	case 10: // overtopping x dir
		h_int = h_init_overtopping(bcs, z_int, x_int);
		break;
	case 11: // overtopping y dir
		h_int = h_init_overtopping(bcs, z_int, y_int);
		break;
	case 12: // radial dam break
		h_int = h_init_radial(x_int, y_int);
		break;
	case 13: // diff topo
		h_int = C(1.95) - z_int;
		break;
	case 14: // non diff topo
		h_int = C(1.78) - z_int;
		break;
	case 15: // triangle topo
		h_int = (x_int < C(15.5) ) ? C(0.75) : C(0.0);
		break;
	case 16: // triangle topo
		h_int = (y_int < C(15.5) ) ? C(0.75) : C(0.0);
		break;
	case 17: // parabolic bowl
		h_int = h_init_parabolic_bowl(x_int, z_int);
		break;
	case 18: // parabolic bowl
		h_int = h_init_parabolic_bowl(y_int, z_int);
		break;
	default:
		break;
	}

	return h_int;
}