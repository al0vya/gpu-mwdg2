#include "write_cumulative_data.h"

void write_cumulative_data
(
	const clock_t           start,
	const real&             time_now,
	const real&             dt,
	const int&              num_cells,
	const SimulationParams& sim_params,
	const char*             respath,
	const bool              first_t_step
)
{
	char fullpath[255];

	sprintf(fullpath, "%s%s", respath, "cumulative-data.csv");
	
	FILE* cumulative_input = fopen(fullpath, (first_t_step) ? "w" : "a");

	if (NULL == cumulative_input)
	{
		fprintf(stderr, "Error in opening cumulative simulation data file.");
		exit(-1);
	}

	if (first_t_step) fprintf(cumulative_input, "simtime,runtime,dt,compression\n");

	const clock_t end = clock();

	const real run_time = (real)(end - start) / CLOCKS_PER_SEC;

	const real compression = C(100.0) - C(100.0) * num_cells / (sim_params.xsz * sim_params.ysz);

	fprintf
	(
		cumulative_input,
		 "%" NUM_FRMT
		",%" NUM_FRMT
		",%" NUM_FRMT
		",%d"
		"\n",
		time_now,
		run_time,
		dt,
		compression
	);

	fclose(cumulative_input);
}