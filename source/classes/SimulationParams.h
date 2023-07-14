#pragma once

#include "../types/Coordinate.h"
#include "../types/real.h"

typedef struct SimulationParams
{
	real       xmin      = C(0.0);
	real       xmax      = C(0.0);
	real       ymin      = C(0.0);
	real       ymax      = C(0.0);
	Coordinate xsz       = 0;
	Coordinate ysz       = 0;
	real       g         = C(9.80665);
	real       time      = C(0.0);
	real       manning   = C(0.0);
	bool       is_monai  = false;
	bool       is_oregon = false;

} SimulationParams;