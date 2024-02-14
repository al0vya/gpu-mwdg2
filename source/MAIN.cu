#include "run_simulation.cuh"
#include "unittests/run_unit_tests.h"

int main
(
	int    argc,
	char** argv
)
{
	#if _RUN_UNIT_TESTS
	run_unit_tests();
	#endif
	
	if (argc < 2)
	{
		fprintf(stderr, "\nNo parameter file specified in command line. Exiting.\n");
		exit(-1);
	}

	run_simulation(argc, argv);

	return 0;
}