#pragma once

#include "../types/Coordinate.h"
#include "../types/real.h"

typedef struct SimulationParams
{
	real       xmin;
	real       xmax;
	real       ymin;
	real       ymax;
	Coordinate xsz;
	Coordinate ysz;
	real       g;
	real       time;
	real       manning;
	bool       is_monai  = false;
	bool       is_oregon = false;

} SimulationParams;