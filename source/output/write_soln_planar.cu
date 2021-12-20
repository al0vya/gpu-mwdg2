#pragma once

#include "write_soln_planar_fv1.cuh"
#include "write_soln_planar_dg2.cuh"

void write_soln_planar
(
    const char*              respath,
	const AssembledSolution& d_assem_sol,
	      real*              d_dt_CFL,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
)
{
	if (solver_params.solver_type == HWFV1)
	{
		write_soln_planar_fv1
		(
			respath,
			d_assem_sol,
			d_dt_CFL,
			dx_finest,
			dy_finest,
			sim_params,
			solver_params,
			saveint
		);
	}
	else if (solver_params.solver_type == MWDG2)
	{
		write_soln_planar_dg2
		(
			respath,
			d_assem_sol,
			d_dt_CFL,
			dx_finest,
			dy_finest,
			sim_params,
			solver_params,
			saveint
		);
	}
}