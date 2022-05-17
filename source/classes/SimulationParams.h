#pragma once

#include "Coordinate.h"
#include "real.h"

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
	bool       is_monai = false;

} SimulationParams;