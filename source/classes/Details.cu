#include "Details.h"

Details::Details
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

Details::Details
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

void Details::write_to_file
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

real Details::verify
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		const real error_eta0 = this->eta0.verify(dirroot, prefix, "eta0-hw");
		const real error_qx0  = this->qx0.verify(dirroot, prefix, "qx0-hw");
		const real error_qy0  = this->qy0.verify(dirroot, prefix, "qy0-hw");
		const real error_z0   = this->z0.verify(dirroot, prefix, "z0-hw");

		return (error_eta0 + error_qx0 + error_qy0 + error_z0) / C(4.0);
	}
	else if (this->solver_type == MWDG2)
	{
		const real error_eta0 = this->eta0.verify(dirroot, prefix, "eta0-mw");
		const real error_qx0  = this->qx0.verify(dirroot, prefix, "qx0-mw");
		const real error_qy0  = this->qy0.verify(dirroot, prefix, "qy0-mw");
		const real error_z0   = this->z0.verify(dirroot, prefix, "z0-mw");

		const real error_eta1x = this->eta1x.verify(dirroot, prefix, "eta1x-mw");
		const real error_qx1x  = this->qx1x.verify(dirroot, prefix, "qx1x-mw");
		const real error_qy1x  = this->qy1x.verify(dirroot, prefix, "qy1x-mw");
		const real error_z1x   = this->z1x.verify(dirroot, prefix, "z1x-mw");

		const real error_eta1y = this->eta1y.verify(dirroot, prefix, "eta1y-mw");
		const real error_qx1y  = this->qx1y.verify(dirroot, prefix, "qx1y-mw");
		const real error_qy1y  = this->qy1y.verify(dirroot, prefix, "qy1y-mw");
		const real error_z1y   = this->z1y.verify(dirroot, prefix, "z1y-mw");

		const real error_0  = (error_eta0  + error_qx0  + error_qy0  + error_z0 ) / C(4.0);
		const real error_1x = (error_eta1x + error_qx1x + error_qy1x + error_z1x) / C(4.0);
		const real error_1y = (error_eta1y + error_qx1y + error_qy1y + error_z1y) / C(4.0);

		return (error_0 + error_1x + error_1y) / C(3.0);
	}
	else
	{
		return C(-999.0);
	}
}