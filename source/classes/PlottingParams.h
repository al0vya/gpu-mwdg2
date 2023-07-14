#pragma once

typedef struct PlottingParams
{
	bool vtk           = false;
	bool c_prop        = false;
	bool raster_out    = false;
	bool voutput_stage = false;
	bool cumulative    = false;
	char dirroot[128]  = {'\0'};
	char resroot[128]  = {'\0'};

} PlottingParams;