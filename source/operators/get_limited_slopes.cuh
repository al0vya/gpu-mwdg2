#pragma once

#include "Slopes.h"
#include "PlanarCoefficients.h"
#include "SolverParams.h"

#include "eval_loc_face_val_dg2.cuh"
#include "minmod.cuh"

__device__
Slopes get_limited_slopes
(
	const PlanarCoefficients& u,
	const PlanarCoefficients& u_n,
	const PlanarCoefficients& u_e,
	const PlanarCoefficients& u_s,
	const PlanarCoefficients& u_w,
	const LegendreBasis&      basis_n,
	const LegendreBasis&      basis_e,
	const LegendreBasis&      basis_s,
	const LegendreBasis&      basis_w,
	const real&               dx_finest,
	const real&               dy_finest,
	const real&               tol_Krivo
);