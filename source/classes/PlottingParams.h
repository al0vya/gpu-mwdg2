#pragma once

typedef struct PlottingParams
{
	bool row_major = false;
	bool vtk       = false;
	bool c_prop    = false;
	bool raster    = false;

	bool any()
	{
		return row_major || vtk || c_prop || raster;
	}

} PlottingParams;