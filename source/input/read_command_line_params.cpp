#include "read_command_line_params.h"

void read_command_line_params
(
	const int&        argc, 
	char**            argv,
	SimulationParams& sim_params,
	SolverParams&     solver_params,
	PlottingParams&   plot_params
)
{
	for (int i = 0; i < argc - 1; i++)
	{
		char* flag  = argv[i];
		char* value = argv[i+1];

		if ( !strncmp(flag, "-epsilon", 8) )
		{
			sscanf(value, "%" NUM_FRMT, &solver_params.epsilon);
		}
		else if ( !strncmp(flag, "-solver", 7) )
		{
			if ( !strncmp(value, "hwfv1", 5) )
			{
				solver_params.solver_type = HWFV1;
				solver_params.CFL         = C(0.5);
			}
			else if ( !strncmp(value, "mwdg2", 5) )
			{
				solver_params.solver_type = MWDG2;
				solver_params.CFL         = C(0.3);
			}
			else
			{
				fprintf(stderr, "Error: invalid adaptive solver type specified in command line, please specify either \"hwfv1\" or \"mwdg2\".");
				exit(-1);
			}
		}
		else if ( !strncmp(flag, "-dirroot", 8) )
		{
			int j = 0; while (plot_params.dirroot[j] != '\0') { plot_params.dirroot[j] = '\0'; j++; }
			
			sscanf(value, "%s", plot_params.dirroot);

			plot_params.make_output_directory();
		}
	}
}