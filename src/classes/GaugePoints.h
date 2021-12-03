#pragma once

#include "SimulationParameters.h"

#include "generate_morton_code.cuh"

typedef struct GaugePoints
{
	MortonCode* codes;
	int         num_points;
	bool        is_copy = false;

	GaugePoints
	(
		const char*                input_filename,
		const SimulationParameters sim_params,
		const real&                cell_size

	)
	{
		num_points = read_num_gauge_points(input_filename);
		codes      = (num_points > 0) ? new MortonCode[num_points] : nullptr;

		read_gauge_points
		(
			input_filename,
			cell_size,
			sim_params
		);
	}

	GaugePoints(const GaugePoints& original) { *this = original; is_copy = true; }

	~GaugePoints() { if (!is_copy && codes != nullptr) delete[] codes; }

	int read_num_gauge_points(const char* input_filename)
	{
		int num_gauge_points = 0;
		
		char str[255]          = {'\0'};
		char buf[64]           = {'\0'};
		char stagefilename[64] = {'\0'};

		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for reading stage file name, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "stagefile", 9) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No stage file name found, proceeding without counting number of stage points.\n");
				fclose(fp);
				return 0;
			}

			sscanf(str, "%s %s", buf, stagefilename);
		}

		fclose(fp);

		fp = fopen(stagefilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening stage file for reading number of stage points, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		fscanf(fp, "%d", &num_gauge_points);

		return num_gauge_points;
	}

	void read_gauge_points
	(
		const char*                 input_filename,
		const real&                 cell_size,
		const SimulationParameters& sim_params
	)
	{
		if (num_points == 0) return;
		
		int num_gauge_points = 0;

		char str[255]          = {'\0'};
		char buf[64]           = {'\0'};
		char stagefilename[64] = {'\0'};

		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for reading stage file name, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "stagefile", 9) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No stage file name found, proceeding without any stage points.\n");
				fclose(fp);
				return;
			}

			sscanf(str, "%s %s", buf, stagefilename);
		}

		fclose(fp);

		fp = fopen(stagefilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening stage file for reading stage points, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		fscanf(fp, "%d", &num_gauge_points);

		real x_stage = C(0.0);
		real y_stage = C(0.0);

		for (int i = 0; i < num_points; i++)
		{
			fscanf(fp, "%" NUM_FRMT " %" NUM_FRMT, &x_stage, &y_stage);

			Coordinate x = (x_stage - sim_params.xmin) / cell_size;
			Coordinate y = (y_stage - sim_params.ymin) / cell_size;

			codes[i] = generate_morton_code(x, y);
		}

		fclose(fp);
	}

} GaugePoints;