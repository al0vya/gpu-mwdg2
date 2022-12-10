#include "read_plot_params.h"

PlottingParams read_plot_params
(
	const char* input_filename
)
{
	PlottingParams plot_params = PlottingParams();

	plot_params.vtk           = read_keyword_str(input_filename, "vtk", 3);
	plot_params.c_prop        = read_keyword_str(input_filename, "c_prop", 6);
	plot_params.raster_out    = read_keyword_str(input_filename, "raster_out", 10);
	plot_params.voutput_stage = read_keyword_str(input_filename, "voutput_stage", 13);
	plot_params.cumulative    = read_keyword_str(input_filename, "cumulative", 10);
	
	return plot_params;
}