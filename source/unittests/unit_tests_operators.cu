#include "unit_tests_operators.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed %s!\n", __func__); } else { printf("Failed %s.\n", __func__); }

void unit_test_dg2_update_RK1_TIMESTEP_1()
{
	const std::string dirroot        = "unittestdata";
	const std::string prefix         = "unit_test_dg2_update_RK1_TIMESTEP_1";
	const std::string par_file       = "unit_tests_MW.par";
	const std::string input_filename = dirroot + "/" + par_file;

	SolverParams      solver_params( input_filename.c_str() );
	Neighbours        d_neighbours( solver_params, dirroot.c_str(), (prefix + "-input").c_str() );
	AssembledSolution d_assem_sol( solver_params, dirroot.c_str(), (prefix + "-input").c_str() ); d_assem_sol.length = 15946;
	AssembledSolution d_buf_assem_sol( solver_params, "buf", dirroot.c_str(), (prefix + "-input").c_str() ); d_buf_assem_sol.length = 15946;
	const int         test_case = 0;
	SimulationParams  sim_params(test_case, input_filename.c_str(), solver_params.L);
	const real        dx_finest = C(0.014);
	const real        dy_finest = C(0.014);
	const real        dt = C(0.001);
	real*             d_dt_CFL = read_d_array_real(d_assem_sol.length, dirroot.c_str(), (prefix + "-input-dt-CFL").c_str() );
	const int&        num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	const bool        rkdg2 = false;

	dg2_update_x<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_assem_sol, 
		d_buf_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);
	
	dg2_update_y<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_assem_sol, 
		d_buf_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);

	const real max_error = C(2e-5);
	const int  max_diffs = 5;

	const real error_neighbours    = d_neighbours.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_assem_sol     = d_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_buf_assem_sol = d_buf_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	
	const int diffs_neighbours    = d_neighbours.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_assem_sol     = d_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_buf_assem_sol = d_buf_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );

	const bool passed =
	(
		error_neighbours    < max_error &&
		error_assem_sol     < max_error &&
		error_buf_assem_sol < max_error &&
		diffs_neighbours    < max_diffs &&
		diffs_assem_sol     < max_diffs &&
		diffs_buf_assem_sol < max_diffs
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_dg2_update_RK2_TIMESTEP_1()
{
	const std::string dirroot        = "unittestdata";
	const std::string prefix         = "unit_test_dg2_update_RK2_TIMESTEP_1";
	const std::string par_file       = "unit_tests_MW.par";
	const std::string input_filename = dirroot + "/" + par_file;

	SolverParams      solver_params( input_filename.c_str() );
	Neighbours        d_neighbours( solver_params, dirroot.c_str(), (prefix + "-input").c_str() );
	AssembledSolution d_buf_assem_sol( solver_params, "buf", dirroot.c_str(), (prefix + "-input").c_str() ); d_buf_assem_sol.length = 15946;
	AssembledSolution d_assem_sol( solver_params, dirroot.c_str(), (prefix + "-input").c_str() ); d_assem_sol.length = 15946;
	const int         test_case = 0;
	SimulationParams  sim_params(test_case, input_filename.c_str(), solver_params.L);
	const real        dx_finest = C(0.014);
	const real        dy_finest = C(0.014);
	const real        dt = C(0.001);
	real*             d_dt_CFL = read_d_array_real(d_assem_sol.length, dirroot.c_str(), (prefix + "-input-dt-CFL").c_str() );
	const int&        num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	const bool        rkdg2 = true;

	dg2_update_x<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_buf_assem_sol, 
		d_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);
	
	dg2_update_y<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_buf_assem_sol, 
		d_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);

	const real max_error = C(2e-5);
	const int  max_diffs = 5;

	const real error_neighbours    = d_neighbours.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_buf_assem_sol = d_buf_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_assem_sol     = d_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	
	const int diffs_neighbours    = d_neighbours.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_buf_assem_sol = d_buf_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_assem_sol     = d_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );

	const bool passed =
	(
		error_neighbours    < max_error &&
		error_buf_assem_sol < max_error &&
		error_assem_sol     < max_error &&
		diffs_neighbours    < max_diffs &&
		diffs_buf_assem_sol < max_diffs &&
		diffs_assem_sol     < max_diffs 
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_dg2_update_RK1_TIMESTEP_2()
{
	const std::string dirroot        = "unittestdata";
	const std::string prefix         = "unit_test_dg2_update_RK1_TIMESTEP_2";
	const std::string par_file       = "unit_tests_MW.par";
	const std::string input_filename = dirroot + "/" + par_file;

	SolverParams      solver_params( input_filename.c_str() );
	Neighbours        d_neighbours( solver_params, dirroot.c_str(), (prefix + "-input").c_str() );
	AssembledSolution d_assem_sol( solver_params, dirroot.c_str(), (prefix + "-input").c_str() ); d_assem_sol.length = 15946;
	AssembledSolution d_buf_assem_sol( solver_params, "buf", dirroot.c_str(), (prefix + "-input").c_str() ); d_buf_assem_sol.length = 15946;
	const int         test_case = 0;
	SimulationParams  sim_params(test_case, input_filename.c_str(), solver_params.L);
	const real        dx_finest = C(0.014);
	const real        dy_finest = C(0.014);
	const real        dt = C(0.003655);
	real*             d_dt_CFL = read_d_array_real(d_assem_sol.length, dirroot.c_str(), (prefix + "-input-dt-CFL").c_str() );
	const int&        num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	const bool        rkdg2 = false;
	
	dg2_update_x<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_assem_sol, 
		d_buf_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);
	
	dg2_update_y<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_assem_sol, 
		d_buf_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);

	const real max_error = C(2e-5);
	const int  max_diffs = 5;

	const real error_assem_sol     = d_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_buf_assem_sol = d_buf_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_neighbours    = d_neighbours.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	
	const int diffs_assem_sol     = d_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_buf_assem_sol = d_buf_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_neighbours    = d_neighbours.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );

	const bool passed =
	(
		error_assem_sol     < max_error &&
		error_buf_assem_sol < max_error &&
		error_neighbours    < max_error &&
		diffs_assem_sol     < max_diffs &&
		diffs_buf_assem_sol < max_diffs &&
		diffs_neighbours    < max_diffs
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}


void unit_test_dg2_update_RK2_TIMESTEP_2()
{
	const std::string dirroot        = "unittestdata";
	const std::string prefix         = "unit_test_dg2_update_RK2_TIMESTEP_2";
	const std::string par_file       = "unit_tests_MW.par";
	const std::string input_filename = dirroot + "/" + par_file;

	SolverParams      solver_params( input_filename.c_str() );
	Neighbours        d_neighbours( solver_params, dirroot.c_str(), (prefix + "-input").c_str() );
	AssembledSolution d_buf_assem_sol( solver_params, "buf", dirroot.c_str(), (prefix + "-input").c_str() ); d_buf_assem_sol.length = 15946;
	AssembledSolution d_assem_sol( solver_params, dirroot.c_str(), (prefix + "-input").c_str() ); d_assem_sol.length = 15946;
	const int         test_case = 0;
	SimulationParams  sim_params(test_case, input_filename.c_str(), solver_params.L);
	const real        dx_finest = C(0.014);
	const real        dy_finest = C(0.014);
	const real        dt = C(0.001);
	real*             d_dt_CFL = read_d_array_real(d_assem_sol.length, dirroot.c_str(), (prefix + "-input-dt-CFL").c_str() );
	const int&        num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	const bool        rkdg2 = true;

	dg2_update_x<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_buf_assem_sol, 
		d_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);
	
	dg2_update_y<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_neighbours, 
		d_buf_assem_sol, 
		d_assem_sol, 
		solver_params, 
		sim_params, 
		dx_finest, 
		dy_finest, 
		dt, 
		test_case,
		d_dt_CFL,
		rkdg2
	);

	const real max_error = C(2e-5);
	const int  max_diffs = 5;

	const real error_neighbours    = d_neighbours.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_buf_assem_sol = d_buf_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	const real error_assem_sol     = d_assem_sol.verify_real( dirroot.c_str(), (prefix + "-output").c_str() );
	
	const int diffs_neighbours    = d_neighbours.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_buf_assem_sol = d_buf_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );
	const int diffs_assem_sol     = d_assem_sol.verify_int( dirroot.c_str(), (prefix + "-output").c_str() );

	const bool passed =
	(
		error_neighbours    < max_error &&
		error_buf_assem_sol < max_error &&
		error_assem_sol     < max_error &&
		diffs_neighbours    < max_diffs &&
		diffs_buf_assem_sol < max_diffs &&
		diffs_assem_sol     < max_diffs 
	);

	if (passed)
		TEST_MESSAGE_PASSED_ELSE_FAILED
}


void run_unit_tests_operators()
{
	unit_test_dg2_update_RK1_TIMESTEP_1();
	unit_test_dg2_update_RK2_TIMESTEP_1();
	unit_test_dg2_update_RK1_TIMESTEP_2();
	unit_test_dg2_update_RK2_TIMESTEP_2();
}

#endif