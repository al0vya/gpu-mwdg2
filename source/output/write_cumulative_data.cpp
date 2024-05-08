#include "write_cumulative_data.h"

void write_cumulative_data
(
	const real&             current_time,
	const real&             inst_time_mra,
	const real&             cumu_time_mra,
	const real&             inst_time_solver,
	const real&             cumu_time_solver,
	const real&             dt,
	const int&              num_timesteps,
	const int&              num_cells,
	const SimulationParams& sim_params,
	const PlottingParams&   plot_params,
	const bool              first_t_step
)
{
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", plot_params.dirroot, '/', plot_params.resroot, ".cumu");
	
	FILE* cumulative_input = fopen(fullpath, (first_t_step) ? "w" : "a");

	if (NULL == cumulative_input)
	{
		fprintf(stderr, "Error in opening cumulative simulation data file.");
		exit(-1);
	}

	if (first_t_step) fprintf(cumulative_input, "simtime,num_timesteps,dt,num_cells,inst_time_solver,cumu_time_solver,inst_time_mra,cumu_time_mra,runtime_total,\n");

	const int num_cells_uniform = sim_params.xsz * sim_params.ysz;
	const int num_cells_adaptive = (num_cells_uniform < num_cells) ? num_cells_uniform : num_cells;

	const real reduction = (real)(num_cells_uniform - num_cells_adaptive) / num_cells_uniform;

	fprintf
	(
		cumulative_input,
		"%"  NUM_FRMT // simtime
		",%d"         // num_timesteps
		",%" NUM_FRMT // dt
		",%d"         // num_cells
		",%" NUM_FRMT // inst_time_solver
		",%" NUM_FRMT // cumu_time_solver
		",%" NUM_FRMT // inst_time_mra
		",%" NUM_FRMT // cumu_time_mra
		",%" NUM_FRMT // runtime_total
		"\n",
		current_time,
		num_timesteps,
		dt,
		num_cells,
		inst_time_solver,
		cumu_time_solver,
		inst_time_mra,
		cumu_time_mra,
		cumu_time_mra + cumu_time_solver
	);

	fclose(cumulative_input);
}