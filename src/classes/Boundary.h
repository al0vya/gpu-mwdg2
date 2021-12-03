#pragma once

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Directions.h"
#include "Coordinate.h"
#include "MortonCode.h"
#include "InletTypes.h"
#include "SimulationParameters.h"

#include "generate_morton_code.cuh"

typedef struct Boundary
{
	Coordinate  start          = 0;
	Coordinate  end            = 0;
	MortonCode* codes          = nullptr;
	int         bdytype        = CLOSED;
	real        inlet          = C(0.0);
	char        timeseries[32] = {'\0'};
	int         timeseries_len = 0;
	real*       time_data      = nullptr;
	real*       inlet_data     = nullptr;
	int         row            = 0;
	int         direction;
	bool        is_copy        = false;

	Boundary
	(
		const char*                 input_filename,
		const SimulationParameters& sim_params,
		const real&                 cell_size,
		const int                   test_case,
		const int                   direction
	)
	:
		direction(direction)
	{
		const int num_cells = read_num_cells
		(
			input_filename,
			sim_params,
			cell_size,
			test_case
		);
		
		codes = (num_cells > 0) ? new MortonCode[num_cells] : nullptr;

		read_bdy_conds
		(
			input_filename,
			sim_params,
			cell_size,
			test_case
		);
		
		read_time_series(input_filename);
	}

	Boundary(const Boundary& original) { *this = original; is_copy = true; }

	~Boundary()
	{
		if (!is_copy)
		{
			if (codes      != nullptr) delete[] codes;
			if (time_data  != nullptr) delete[] time_data;
			if (inlet_data != nullptr) delete[] inlet_data;
		}
	}

	__device__ __forceinline__
	bool bound(const Coordinate& coordinate) const { return (coordinate >= start && coordinate <= end); }

	__host__ __forceinline__
	int num_cells() const { return end - start + 1; }

	__device__ __forceinline__
	real q_src(const real& dt, const real& dx) { return inlet * dt / dx; }

	int read_num_cells
	(
		const char*                 input_filename,
		const SimulationParameters& sim_params,
		const real&                 cell_size,
		const int&                  test_case
	)
	{
		if (test_case != 0) return 0;
		
		char bcifilename[32] = {'\0'};
		char buf[64]         = {'\0'};
		char str[255]        = {'\0'};
	
		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for reading boundary cells, file: %s, line: %d", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "bcifile", 7) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No boundary condition file found to count cells, proceeding with default (closed) boundary conditions.\n");
				fclose(fp);

				return 0;
			}

			sscanf(str, "%s %s", buf, bcifilename);
		}
	
		fclose(fp);
		
		real upper = C(0.0);
		real lower = C(0.0);

		char bcidir  = '\0';
		char filedir = '\0';

		char bdytype_buf[8] = {'\0'};

		real origin = C(0.0);

		switch (direction)
		{
		case NORTH:
			origin = sim_params.xmin;
			bcidir = 'N';
			break;
		case EAST:
			origin = sim_params.ymin;
			bcidir = 'E';
			break;
		case SOUTH:
			origin = sim_params.xmin;
			bcidir = 'S';
			break;
		case WEST:
			origin = sim_params.ymin;
			bcidir = 'W';
			break;
		default:
			break;
		}
	
		fp = fopen(bcifilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening boundary condition file, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		while (filedir != bcidir)
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No enforced boundary cells counted for boundary %c.\n", bcidir);
				fclose(fp);

				return 0;
			}

			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s", &filedir, &lower, &upper, bdytype_buf);
		}

		fclose(fp);

		int start = (lower - origin) / cell_size;
		int end   = (upper - origin) / cell_size;

		return end - start + 1;
	}

	void gen_bdy_morton_codes
	(
		const SimulationParameters& sim_params
	)
	{
		Coordinate current = 0;
	
		switch (direction)
		{
		case SOUTH:
		{
			for (int i = 0; i < this->num_cells(); i++)
			{
				current = this->start + i;

				this->codes[i] = generate_morton_code(current, 0);
			}

			break;
		}
		case NORTH:
		{
			for (int i = 0; i < this->num_cells(); i++)
			{
				current = this->start + i;

				this->codes[i] = generate_morton_code(current, sim_params.ysz - 1);
			}

			break;
		}
		case EAST:
		{
			for (int i = 0; i < this->num_cells(); i++)
			{
				current = this->start + i;

				this->codes[i] = generate_morton_code(sim_params.xsz - 1, current);
			}

			break;
		}
		case WEST:
		{
			for (int i = 0; i < this->num_cells(); i++)
			{
				current = this->start + i;

				this->codes[i] = generate_morton_code(0, current);
			}

			break;
		}
		default:
			break;
		}
	}

	void read_bdy_conds
	(
		const char*                 input_filename,
		const SimulationParameters& sim_params,
		const real&                 cell_size,
		const int&                  test_case
	)
	{
		if (test_case != 0) return;
		
		char bcifilename[32] = {'\0'};
		char buf[64]         = {'\0'};
		char str[255]        = {'\0'};
	
		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for reading boundary cells, file: %s, line: %d", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "bcifile", 7) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No boundary condition file found to read, proceeding with default (closed) boundary conditions.\n");
				fclose(fp);

				return;
			}

			sscanf(str, "%s %s", buf, bcifilename);
		}
	
		fclose(fp);

		real upper = C(0.0);
		real lower = C(0.0);

		char bcidir  = '\0';
		char filedir = '\0';

		char bdytype_buf[8] = {'\0'};

		real origin = C(0.0);

		switch (direction)
		{
		case NORTH:
			origin = sim_params.xmin;
			bcidir = 'N';
			break;
		case EAST:
			origin = sim_params.ymin;
			bcidir = 'E';
			break;
		case SOUTH:
			origin = sim_params.xmin;
			bcidir = 'S';
			break;
		case WEST:
			origin = sim_params.ymin;
			bcidir = 'W';
			break;
		default:
			break;
		}
	
		fp = fopen(bcifilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening boundary condition file, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}
		
		int  bdytype        = CLOSED;
		real inlet          = C(0.0);
		char timeseries[32] = {'\0'};

		while (filedir != bcidir)
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No specifications found for boundary %c, proceeding with closed boundaries.\n", bcidir);
				fclose(fp);

				return;
			}

			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s", &filedir, &lower, &upper, bdytype_buf);
		}

		fclose(fp);
	
		if ( !strncmp(bdytype_buf, "CLOSED", 6) )
		{
			bdytype = CLOSED;
		}
		else if ( !strncmp(bdytype_buf, "FREE", 4) )
		{
			bdytype = FREE;
		}
		else if ( !strncmp(bdytype_buf, "HFIX", 4) )
		{
			bdytype = HFIX;
			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %" NUM_FRMT, &filedir, &lower, &upper, bdytype_buf, &inlet);
		}
		else if ( !strncmp(bdytype_buf, "HVAR", 4) )
		{
			bdytype = HVAR;
			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %s", &filedir, &lower, &upper, bdytype_buf, timeseries);
		}
		else if ( !strncmp(bdytype_buf, "QFIX", 4) )
		{
			bdytype = QFIX;
			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %" NUM_FRMT, &filedir, &lower, &upper, bdytype_buf, &inlet);
		}
		else if ( !strncmp(bdytype_buf, "QVAR", 4) )
		{
			bdytype = QVAR;
			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %s", &filedir, &lower, &upper, bdytype_buf, timeseries);
		}

		this->start   = (lower - origin) / cell_size;
		this->end     = (upper - origin) / cell_size - 1;
		gen_bdy_morton_codes(sim_params);
		this->bdytype = bdytype;
		this->inlet   = inlet;
		sprintf(this->timeseries, "%s", timeseries);
	}

	void read_time_series
	(
		const char* input_filename
	)
	{
		if ( (bdytype != HVAR && bdytype != QVAR) )
		{
			time_data  = nullptr;
			inlet_data = nullptr;
			
			return;
		}

		char str[255]        = {'\0'};
		char buf[64]         = {'\0'};
		char bdyfilename[64] = {'\0'};

		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for updating boundary inlet, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		while (strncmp(buf, "bdyfile", 7) )
		{
			if (NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stderr, "Error reading time varying boundary conditions, file: %s, line: %d.\n", __FILE__, __LINE__);
				fclose(fp);
				exit(-1);
			}

			sscanf(str, "%s %s", buf, bdyfilename);
		}

		fclose(fp);

		fp = fopen(bdyfilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening time varying boundary condition file %s, file: %s, line: %d.\n", bdyfilename, __FILE__, __LINE__);
			exit(-1);
		}

		char* timeseriesptr = timeseries;

		int num_char_timeseries = 0;

		while (*(timeseriesptr + num_char_timeseries) != '\0') num_char_timeseries++;

		while ( strncmp(buf, timeseries, num_char_timeseries) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stderr, "Error reading boundary inlet time series \"%s\", file: %s, line: %d.\n", timeseries, __FILE__, __LINE__);
				fclose(fp);
				exit(-1);
			}

			sscanf(str, "%s", buf);
		}

		int num_rows_timeseries = 0;

		fgets(str, sizeof(str), fp);

		sscanf(str, "%d %s", &num_rows_timeseries, buf);

		int time_multiplier = ( !strncmp(buf, "seconds", 7) ) ? 1 : ( !strncmp(buf, "minutes", 7) ) ? 60 : 3600;

		timeseries_len = num_rows_timeseries;

		if (num_rows_timeseries == 0)
		{
			fprintf(stderr, "Zero entries for timeseries: \"%s\", file: %s, line: %d.\n", timeseries, __FILE__, __LINE__);
			fclose(fp);
			exit(-1);
		}

		time_data  = (num_rows_timeseries > 0) ? new real[num_rows_timeseries] : nullptr;
		inlet_data = (num_rows_timeseries > 0) ? new real[num_rows_timeseries] : nullptr;

		for (int i = 0; i < num_rows_timeseries; i++)
		{
			fgets(str, sizeof(str), fp);

			sscanf(str, "%" NUM_FRMT " %" NUM_FRMT, &inlet_data[i], &time_data[i]);

			time_data[i] *= time_multiplier;
		}

		fclose(fp);
	}

	void update_inlet
	(
		const real& time_now
	)
	{
		if ( (bdytype != HVAR && bdytype != QVAR) ) return;

		if ( (row - 1) < timeseries_len )
		{
			real t_1 = time_data[row];
			real t_2 = time_data[row + 1];

			if (time_now > t_2)
			{
				row++; 
				
				t_1 = time_data[row];
				t_2 = time_data[row + 1];
			}

			real inlet_1 = inlet_data[row];
			real inlet_2 = inlet_data[row + 1];

			inlet = inlet_1 + (inlet_2 - inlet_1) / (t_2 - t_1) * (time_now - t_1);
		}
		else
		{
			inlet = inlet_data[timeseries_len - 1];
		}
	}

} Boundary;