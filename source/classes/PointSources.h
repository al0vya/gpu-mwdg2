#pragma once

#include "cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "MortonCode.h"
#include "InletTypes.h"
#include "SimulationParams.h"

#include "generate_morton_code.cuh"

typedef struct PointSources
{
	MortonCode* h_codes         = nullptr;
	MortonCode* d_codes         = nullptr;
	real*       h_srcs          = nullptr;
	real*       d_srcs          = nullptr;
	int*        h_src_types     = nullptr;
	int*        d_src_types     = nullptr;
	char*       timeseries      = nullptr;
	int*        timeseries_lens = nullptr;
	real**      all_time_data   = nullptr;
	real**      all_src_data    = nullptr;
	int*        rows            = nullptr;
	int         num_srcs        = 0;
	bool        is_copy         = false;

	PointSources
	(
		const char*                 input_filename,
		const SimulationParams& sim_params,
		const real&                 cell_size,
		const int&                  test_case,
		const real&                 dt
	)
	{
		if (test_case != 0)
		{
			fprintf(stdout, "Running built-in test case without any point sources.\n");
			return;
		}
		
		num_srcs = read_num_point_srcs(input_filename);
		
		size_t bytes_codes     = sizeof(MortonCode) * num_srcs;
		size_t bytes_srcs      = sizeof(real)       * num_srcs;
		size_t bytes_src_types = sizeof(int)        * num_srcs;

		h_codes         = (num_srcs > 0) ? new MortonCode[num_srcs]                : nullptr;
		d_codes         = (num_srcs > 0) ? (MortonCode*)malloc_device(bytes_codes) : nullptr;
		h_srcs          = (num_srcs > 0) ? new real[num_srcs]()                    : nullptr;
		d_srcs          = (num_srcs > 0) ? (real*)malloc_device(bytes_srcs)        : nullptr;
		h_src_types     = (num_srcs > 0) ? new int[num_srcs]                       : nullptr;
		d_src_types     = (num_srcs > 0) ? (int*)malloc_device(bytes_src_types)    : nullptr;
		timeseries      = (num_srcs > 0) ? new char[num_srcs * 32]()               : nullptr;
		timeseries_lens = (num_srcs > 0) ? new int[num_srcs]()                     : nullptr;
		all_time_data   = (num_srcs > 0) ? new real*[num_srcs]()                   : nullptr;
		all_src_data    = (num_srcs > 0) ? new real*[num_srcs]()                   : nullptr;
		rows            = (num_srcs > 0) ? new int[num_srcs]()                     : nullptr;

		read_point_srcs
		(
			input_filename,
			num_srcs,
			sim_params,
			cell_size
		);

		read_all_timeseries(input_filename);

		update_all_srcs(dt);
	}

	PointSources(const PointSources& original) { *this = original; is_copy = true; }

	~PointSources()
	{
		if (!is_copy)
		{
			if (h_codes         != nullptr) delete[] h_codes;
			if (d_codes         != nullptr) CHECK_CUDA_ERROR( free_device(d_codes) );
			if (h_src_types     != nullptr) delete[] h_src_types;
			if (d_src_types     != nullptr) CHECK_CUDA_ERROR( free_device(d_src_types) );
			if (h_srcs          != nullptr) delete[] h_srcs;
			if (d_srcs          != nullptr) CHECK_CUDA_ERROR( free_device(d_srcs) );
			if (timeseries      != nullptr) delete[] timeseries;
			if (timeseries_lens != nullptr) delete[] timeseries_lens;
			
			if (all_time_data != nullptr)
			{
				for (int i = 0; i < num_srcs; i++)
				{
					if (all_time_data[i] != nullptr)
					{
						delete[] all_time_data[i];
					}
				}
			}

			if (all_time_data != nullptr) delete[]all_time_data;

			if (all_src_data != nullptr)
			{
				for (int i = 0; i < num_srcs; i++)
				{
					if (all_src_data[i] != nullptr)
					{
						delete[] all_src_data[i];
					}
				}
			}

			if (all_src_data != nullptr) delete[] all_src_data;

			if (rows != nullptr) delete[] rows;

		}
	}
	
	__device__ __forceinline__
	real q_src(const real& dt, const real& dx, const int idx) { return this->d_srcs[idx] * dt / dx; }

	int read_num_point_srcs
	(
		const char* input_filename
	)
	{
		char bcifilename[32] = {'\0'};
		char buf[128]        = {'\0'};
		char str[255]        = {'\0'};

		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for counting numer of point sources, file: %s, line: %d", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "bcifile", 7) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No point source file found, proceeding without counting number of point sources.\n");
				fclose(fp);

				return 0;
			}

			sscanf(str, "%s %s", buf, bcifilename);
		}

		fclose(fp);

		fp = fopen(bcifilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening point source file for counting number of point sources, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		char point    = '\0';
		int  num_srcs = 0;

		while ( !( NULL == fgets(str, sizeof(str), fp) ) )
		{
			sscanf(str, "%c", &point);

			if (point == 'P') num_srcs++;
		}

		if (num_srcs == 0)
		{
			fprintf(stdout, "No point sources counted in boundary condition file, proceeding with zero point sources.\n");
			fclose(fp);

			return 0;
		}
	
		fclose(fp);

		return num_srcs;
	}

	void read_point_srcs
	(
		const char*                 input_filename,
		const int&                  num_srcs,
		const SimulationParams& sim_params,
		const real&                 cell_size
	)
	{
		if (num_srcs == 0) return;
		
		char bcifilename[32] = {'\0'};
		char buf[128]        = {'\0'};
		char str[255]        = {'\0'};

		FILE* fp = fopen(input_filename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening input file for reading point sources, file: %s, line: %d", __FILE__, __LINE__);
			exit(-1);
		}

		while ( strncmp(buf, "bcifile", 7) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stdout, "No point source file found, proceeding with no point sources.\n");
				fclose(fp);

				return;
			}

			sscanf(str, "%s %s", buf, bcifilename);
		}

		fclose(fp);

		fp = fopen(bcifilename, "r");

		if (NULL == fp)
		{
			fprintf(stderr, "Error opening point source file, file: %s, line: %d.\n", __FILE__, __LINE__);
			exit(-1);
		}

		real x_stage = C(0.0);
		real y_stage = C(0.0);
	
		Coordinate x = 0;
		Coordinate y = 0;

		char point             = '\0';
		char inlet_type_buf[8] = {'\0'};
		int  srcs_counted      = 0;

		while (srcs_counted < num_srcs)
		{
			fgets(str, sizeof(str), fp);

			sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s", &point, &x_stage, &y_stage, inlet_type_buf);

			if (point == 'P')
			{
				x = (x_stage - sim_params.xmin) / cell_size;
				y = (y_stage - sim_params.ymin) / cell_size;

				h_codes[srcs_counted] = generate_morton_code(x, y);

				if ( !strncmp(inlet_type_buf, "HFIX", 4) )
				{
					h_src_types[srcs_counted] = HFIX;
					sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %" NUM_FRMT, &point, &x_stage, &y_stage, inlet_type_buf, &h_srcs[srcs_counted]);
				}
				else if ( !strncmp(inlet_type_buf, "HVAR", 4) )
				{
					h_src_types[srcs_counted] = HVAR;
					sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %s", &point, &x_stage, &y_stage, inlet_type_buf, &timeseries[srcs_counted * 32]);
				}
				else if ( !strncmp(inlet_type_buf, "QFIX", 4) )
				{
					h_src_types[srcs_counted] = QFIX;
					sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %" NUM_FRMT, &point, &x_stage, &y_stage, inlet_type_buf, &h_srcs[srcs_counted]);
				}
				else if ( !strncmp(inlet_type_buf, "QVAR", 4) )
				{
					h_src_types[srcs_counted] = QVAR;
					sscanf(str, "%c %" NUM_FRMT " %" NUM_FRMT " %s %s", &point, &x_stage, &y_stage, inlet_type_buf, &timeseries[srcs_counted * 32]);
				}

				srcs_counted++;
			}
		}

		fclose(fp);
	
		size_t bytes_codes     = sizeof(MortonCode) * num_srcs;
		size_t bytes_srcs      = sizeof(real)       * num_srcs;
		size_t bytes_src_types = sizeof(int)        * num_srcs;

		copy(d_codes,     h_codes,     bytes_codes);
		copy(d_srcs,      h_srcs,      bytes_srcs);
		copy(d_src_types, h_src_types, bytes_src_types);
	}

	void read_timeseries
	(
		const char* input_filename,
		const int&  src_type,
		const char* timeseries,
		int&        timeseries_len,
		real*&      time_data,
		real*&      src_data
	)
	{
		if ( (src_type != HVAR && src_type != QVAR) )
		{
			time_data = nullptr;
			src_data  = nullptr;
			
			return;
		}

		char bdyfilename[64] = {'\0'};
		char str[255]        = {'\0'};
		char buf[128]        = {'\0'};

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
			fprintf(stderr, "Error opening bdyfile %s, file: %s, line: %d.\n", bdyfilename, __FILE__, __LINE__);
			exit(-1);
		}

		int num_char_timeseries = 0;

		while ( *(timeseries + num_char_timeseries) != '\0' ) num_char_timeseries++;

		while ( strncmp(buf, timeseries, num_char_timeseries) )
		{
			if ( NULL == fgets(str, sizeof(str), fp) )
			{
				fprintf(stderr, "Error reading point source time series \"%s\", file: %s, line: %d.\n", timeseries, __FILE__, __LINE__);
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

		time_data = (num_rows_timeseries > 0) ? new real[num_rows_timeseries] : nullptr;
		src_data  = (num_rows_timeseries > 0) ? new real[num_rows_timeseries] : nullptr;

		real src  = C(0.0);
		real time = C(0.0);

		for (int i = 0; i < num_rows_timeseries; i++)
		{
			fgets(str, sizeof(str), fp);

			sscanf( str, "%" NUM_FRMT " %" NUM_FRMT, &src, &time );

			src_data[i]  = src;
			time_data[i] = time * time_multiplier;
		}

		fclose(fp);
	}

	void read_all_timeseries(const char* input_filename)
	{
		for (int i = 0; i < this->num_srcs; i++)
		{
			read_timeseries
			(
				input_filename, 
				this->h_src_types[i],
			    &this->timeseries[i * 32],
			    this->timeseries_lens[i],
			    this->all_time_data[i],
			    this->all_src_data[i]
			);
		}
	}

	void update_src
	(
		const real& time_now,
		const int&  src_type,
		real&       h_src,
		int&        row,
		const int&  timeseries_len,
		real*       time_data,
		real*       src_data
	)
	{
		if ( (src_type != HVAR && src_type != QVAR) ) return;

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

			real src_1 = src_data[row];
			real src_2 = src_data[row + 1];

			h_src = src_1 + (src_2 - src_1) / (t_2 - t_1) * (time_now - t_1);
		}
		else
		{
			h_src = src_data[timeseries_len - 1];
		}
	}

	void update_all_srcs(const real& time_now)
	{
		for (int i = 0; i < this->num_srcs; i++)
		{
			update_src
			(
				time_now,
				this->h_src_types[i],
			    this->h_srcs[i],
			    this->rows[i],
			    this->timeseries_lens[i],
			    this->all_time_data[i],
			    this->all_src_data[i]
			);
;		}

		size_t bytes_srcs = sizeof(real) * num_srcs;

		copy(d_srcs, h_srcs, bytes_srcs);
	}

} PointSources;