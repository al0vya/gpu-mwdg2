#include "run_unit_tests.cuh"

#if _RUN_UNIT_TESTS

void run_unit_tests()
{
	run_unit_tests_classes();
	run_unit_tests_output();
	run_unit_tests_input();
	run_unit_tests_mra();
	run_unit_tests_utilities();
}

#endif