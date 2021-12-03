#include "write_cumu_sim_time.h"

void write_cumu_sim_time
(
	const clock_t              start,
	const real                 time_now,
	const char*                respath,
	const bool                 first_t_step
)
{
	char fullpath[255];

	sprintf(fullpath, "%s%s", respath, "clock_time_vs_sim_time.csv");
	
	FILE* cumu_sim_time_input = (first_t_step) ? fopen(fullpath, "w") : fopen(fullpath, "a");

	if (NULL == cumu_sim_time_input)
	{
		fprintf(stderr, "Error in opening cumulative simulation time file.");
		exit(-1);
	}

	if (first_t_step) fprintf(cumu_sim_time_input, "sim_time,clock_time\n");

	clock_t end = clock();

	real run_time = (real)(end - start) / CLOCKS_PER_SEC;

	fprintf(cumu_sim_time_input, "%" NUM_FRMT ",%" NUM_FRMT "\n", time_now, run_time);

	fclose(cumu_sim_time_input);
}