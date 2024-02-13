#pragma once

#include "../input/read_keyword_bool.h"
#include "../input/read_keyword_str.h"

typedef struct PlottingParams
{
	bool vtk           = false;
	bool c_prop        = false;
	bool raster_out    = false;
	bool voutput_stage = false;
	bool voutput       = false;
	bool qoutput       = false;
	bool elevoff       = false;
	bool depthoff      = false;
	bool cumulative    = false;
	char dirroot[128]  = {'\0'};
	char resroot[128]  = {'\0'};

	PlottingParams
	(
		const char* input_filename
	)
	{
		this->vtk           = read_keyword_bool(input_filename, "vtk", 3);
		this->c_prop        = read_keyword_bool(input_filename, "c_prop", 6);
		this->raster_out    = read_keyword_bool(input_filename, "raster_out", 10);
		this->voutput_stage = read_keyword_bool(input_filename, "voutput_stage", 13);
		this->voutput       = read_keyword_bool(input_filename, "voutput", 7);
		this->qoutput       = read_keyword_bool(input_filename, "qoutput", 7);
		this->elevoff       = read_keyword_bool(input_filename, "elevoff", 7);
		this->depthoff      = read_keyword_bool(input_filename, "depthoff", 7);
		this->cumulative    = read_keyword_bool(input_filename, "cumulative", 10);
		
		read_keyword_str(input_filename, "resroot", 7, this->resroot);
		read_keyword_str(input_filename, "dirroot", 7, this->dirroot);

		// if no result filename prefix is specified
		if (this->resroot[0] == '\0')
		{
			// default prefix is "res"
			sprintf(this->resroot, "%s", "res");
		}
		
		// if no results folder name is specified
		if (this->dirroot[0] == '\0')
		{
			// results folder name is "res"
			sprintf(this->dirroot, "%s", "res");
		}

		make_output_directory();
	}

	void make_output_directory()
	{
		char sys_cmd_str_buf[255] = {'\0'};
		sprintf(sys_cmd_str_buf, "%s %s", "mkdir", this->dirroot);
		system(sys_cmd_str_buf);
	}

} PlottingParams;