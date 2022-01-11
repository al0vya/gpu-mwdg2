#include "read_plot_params.h"

PlottingParams read_plot_params
(
	const char* input_filename
)
{
	PlottingParams plot_params = PlottingParams();

	char plot_buf[128] = {'\0'};

	read_keyword_str(input_filename, "row_major", 9, plot_buf);
	plot_params.row_major = ( !strncmp(plot_buf, "on", 2) );
	
	read_keyword_str(input_filename, "planar", 6, plot_buf);
	plot_params.planar = ( !strncmp(plot_buf, "on", 2) );
	
	read_keyword_str(input_filename, "vtk", 3, plot_buf);
	plot_params.vtk = ( !strncmp(plot_buf, "on", 2) );
	
	read_keyword_str(input_filename, "c_prop", 6, plot_buf);
	plot_params.c_prop = ( !strncmp(plot_buf, "on", 2) );
	
	read_keyword_str(input_filename, "raster", 6, plot_buf);
	plot_params.raster = ( !strncmp(plot_buf, "on", 2) );
	
	read_keyword_str(input_filename, "voutput_stage", 13, plot_buf);
	plot_params.voutput_stage = ( !strncmp(plot_buf, "on", 2) );
	
	return plot_params;
}