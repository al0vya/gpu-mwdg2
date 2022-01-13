#pragma once

typedef struct PlottingParams
{
	bool row_major     = false;
	bool planar        = false;
	bool vtk           = false;
	bool c_prop        = false;
	bool raster_out    = false;
	bool voutput_stage = false;

} PlottingParams;