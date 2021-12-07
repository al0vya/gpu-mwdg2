#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <math.h>

#include "real.h"

void read_raster_file
(
	const char* raster_filename,
	real*       raster_array,
	const int&  mesh_dim,
	const real& wall_height
);