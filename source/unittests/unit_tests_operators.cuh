#pragma once

#include "../operators/dg2_update.cuh"
#include "../operators/dg2_update_x.cuh"
#include "../operators/dg2_update_y.cuh"

void generate_data_unit_test_dg2_update_RK1
(
	const char*        dirroot,
	const char*        input_or_output_str,
	Neighbours&        d_neighbours,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	const real&        dx_finest,
	const real&        dy_finest,
	const real&        dt,
	real*              d_dt_CFL,
	const int&         timestep
);

void run_unit_tests_operators();