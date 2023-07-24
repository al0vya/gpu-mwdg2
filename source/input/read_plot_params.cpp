#include "read_plot_params.h"

PlottingParams read_plot_params
(
	const char* input_filename
)
{
	PlottingParams plot_params = PlottingParams();

	plot_params.vtk           = read_keyword_bool(input_filename, "vtk", 3);
	plot_params.c_prop        = read_keyword_bool(input_filename, "c_prop", 6);
	plot_params.raster_out    = read_keyword_bool(input_filename, "raster_out", 10);
	plot_params.voutput_stage = read_keyword_bool(input_filename, "voutput_stage", 13);
	plot_params.cumulative    = read_keyword_bool(input_filename, "cumulative", 10);
	
	read_keyword_str(input_filename, "resroot", 7, plot_params.resroot);
	read_keyword_str(input_filename, "dirroot", 7, plot_params.dirroot);

	// if no result filename prefix is specified
	if (plot_params.resroot[0] == '\0')
	{
		// default prefix is "res"
		sprintf(plot_params.resroot, "%s", "res");
	}
    
    // if no results folder name is specified
	if (plot_params.dirroot[0] == '\0')
	{
		// results folder name is "res"
		sprintf(plot_params.dirroot, "%s", "res");
	}

	char sys_cmd_str_buf[255] = {'\0'};
	sprintf(sys_cmd_str_buf, "%s%s", "mkdir ", plot_params.dirroot);
	system(sys_cmd_str_buf);

	return plot_params;
}