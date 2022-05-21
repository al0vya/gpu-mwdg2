#include "write_cumulative_data.h"

void write_cumulative_data
(
	const clock_t start,
	const real&   time_now,
	const real&   dt,
	const int&    num_cells,
	const char*   respath,
	const bool    first_t_step
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

	if (first_t_step) fprintf(cumulative_input, "simtime,runtime,dt,cells\n");

	clock_t end = clock();

	real run_time = (real)(end - start) / CLOCKS_PER_SEC;

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
		num_cells
	);

	fclose(cumulative_input);
}