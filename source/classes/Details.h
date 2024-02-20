#pragma once

#include "SubDetails.h"
#include "../types/SolverTypes.h"

typedef struct Details
{
	SubDetails eta0;
	SubDetails qx0;
	SubDetails qy0;
	SubDetails z0;

	SubDetails eta1x;
	SubDetails qx1x;
	SubDetails qy1x;
	SubDetails z1x;

	SubDetails eta1y;
	SubDetails qx1y;
	SubDetails qy1y;
	SubDetails z1y;

	int solver_type = 0;

	Details
	(
		const SolverParams& solver_params
	)
	:
		eta0(solver_params),
		qx0 (solver_params),
		qy0 (solver_params),
		z0  (solver_params),

		eta1x( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qx1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qy1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		z1x  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		
		eta1y( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qx1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qy1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		z1y  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),

		solver_type(solver_params.solver_type)
	{}

	Details
	(
		const SolverParams& solver_params,
		const char*         dirroot
	)
	:
		eta0( solver_params, dirroot, (solver_params.solver_type == MWDG2) ? "eta0-mw" : "eta0-hw"),
		qx0 ( solver_params, dirroot, (solver_params.solver_type == MWDG2) ? "qx0-mw"  : "qx0-hw"),
		qy0 ( solver_params, dirroot, (solver_params.solver_type == MWDG2) ? "qy0-mw"  : "qy0-hw"),
		z0  ( solver_params, dirroot, (solver_params.solver_type == MWDG2) ? "z0-mw"   : "z0-hw"),
		
		eta1x( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "eta1x-mw"),
		qx1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "qx1x-mw" ),
		qy1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "qy1x-mw" ),
		z1x  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "z1x-mw"  ),
		
		eta1y( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "eta1y-mw"),
		qx1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "qx1y-mw" ),
		qy1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "qy1y-mw" ),
		z1y  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams(), dirroot, "z1y-mw"  ),
		
		solver_type(solver_params.solver_type)
	{}

	void write_to_file
	(
		const char* dirroot,
		const char* prefix
	)
	{
		if (this->solver_type == HWFV1)
		{
			eta0.write_to_file(dirroot, prefix, "eta0-hw");
			qx0.write_to_file(dirroot,  prefix, "qx0-hw");
			qy0.write_to_file(dirroot,  prefix, "qy0-hw");
			z0.write_to_file(dirroot,   prefix, "z0-hw");
		}
		else if (this->solver_type == MWDG2)
		{
			eta0.write_to_file(dirroot, prefix, "eta0-mw");
			qx0.write_to_file(dirroot,  prefix, "qx0-mw");
			qy0.write_to_file(dirroot,  prefix, "qy0-mw");
			z0.write_to_file(dirroot,   prefix, "z0-mw");

			eta1x.write_to_file(dirroot, prefix, "eta1x-mw");
			qx1x.write_to_file(dirroot,  prefix, "qx1x-mw");
			qy1x.write_to_file(dirroot,  prefix, "qy1x-mw");
			z1x.write_to_file(dirroot,   prefix, "z1x-mw");

			eta1y.write_to_file(dirroot, prefix, "eta1y-mw");
			qx1y.write_to_file(dirroot,  prefix, "qx1y-mw");
			qy1y.write_to_file(dirroot,  prefix, "qy1y-mw");
			z1y.write_to_file(dirroot,   prefix, "z1y-mw");
		}
	}

} Details;