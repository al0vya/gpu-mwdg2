#include "run_simulation.cuh"

int main
(
	int    argc,
	char** argv
)
{
	#if _RUN_UNIT_TESTS
	if (argc == 1)
	{
		run_unit_tests();
	}
	#endif
	
	if (argc < 2)
	{
		printf("\nNo parameter file specified in command line. Exiting.\n");
		return 0;
	}

	run_simulation(argc, argv);

	return 0;
}