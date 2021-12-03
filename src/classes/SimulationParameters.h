#pragma once

#include "Coordinate.h"
#include "real.h"

typedef struct SimulationParameters
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

} SimulationParameters;