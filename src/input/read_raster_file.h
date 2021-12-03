#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "real.h"

void read_raster_file
(
	const char* raster_filename,
	real*       raster_array,
	const int&  mesh_dim,
	const real& wall_height
);