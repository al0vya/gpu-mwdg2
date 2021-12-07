#pragma once

#include "cuda_runtime.h"

#include "FlowVector.h"
#include "SolverParams.h"
#include "SimulationParams.h"

__device__ __forceinline__
FlowVector flux_HLL_x
(
	const FlowVector&           U_neg,
	const FlowVector&           U_pos,
	const SolverParams&     solver_params,
	const SimulationParams& sim_params
)
{
	if (U_neg.h < solver_params.tol_h && U_pos.h < solver_params.tol_h) return FlowVector();

	const real h_neg = (U_neg.h < solver_params.tol_h) ? C(0.0) : U_neg.h;
	const real u_neg = (U_neg.h < solver_params.tol_h) ? C(0.0) : U_neg.qx / U_neg.h;
	const real v_neg = (U_neg.h < solver_params.tol_h) ? C(0.0) : U_neg.qy / U_neg.h;
	
	const real h_pos = (U_pos.h < solver_params.tol_h) ? C(0.0) : U_pos.h;
	const real u_pos = (U_pos.h < solver_params.tol_h) ? C(0.0) : U_pos.qx / U_pos.h;
	const real v_pos = (U_pos.h < solver_params.tol_h) ? C(0.0) : U_pos.qy / U_pos.h;
	
	const real a_neg = sqrt(sim_params.g * h_neg);
	const real a_pos = sqrt(sim_params.g * h_pos);
	
	const real h_star = 
		( C(0.5) * (a_neg + a_pos) + C(0.25) * (u_neg - u_pos) ) * ( C(0.5) * (a_neg + a_pos) + C(0.25) * (u_neg - u_pos) )
		/
		sim_params.g;

	const real u_star = C(0.5) * (u_neg + u_pos) + a_neg - a_pos;
	
	const real a_star = sqrt(sim_params.g * h_star);
	
	const real s_neg = (h_neg < solver_params.tol_h) ? u_pos - C(2.0) * a_pos : min(u_neg - a_neg, u_star - a_star);
	const real s_pos = (h_pos < solver_params.tol_h) ? u_neg + C(2.0) * a_neg : max(u_pos + a_pos, u_star + a_star);
	
	const FlowVector F_neg =
	{
		U_neg.qx,
		u_neg * U_neg.qx + C(0.5) * sim_params.g * h_neg * h_neg,
		U_neg.qy * u_neg
	};

	const FlowVector F_pos =
	{
		U_pos.qx,
		u_pos * U_pos.qx + C(0.5) * sim_params.g * h_pos * h_pos,
		U_pos.qy * u_pos
	};

	if ( s_neg >= C(0.0) )
	{
		return F_neg;
	}
	else if ( s_neg < C(0.0) && s_pos >= C(0.0) )
	{
		FlowVector F = {};

		F.h =
			( s_pos * F_neg.h - s_neg * F_pos.h + s_neg * s_pos * (h_pos - h_neg) )
			/
			(s_pos - s_neg);

		F.qx =
			( s_pos * F_neg.qx - s_neg * F_pos.qx + s_neg * s_pos * (U_pos.qx - U_neg.qx) )
			/
			(s_pos - s_neg);

		const real s_mid =
			( s_neg * h_pos * (u_pos - s_pos) - s_pos * h_neg * (u_neg - s_neg) )
			/
			( h_pos * (u_pos - s_pos) - h_neg * (u_neg - s_neg) );

		F.qy = ( s_neg < C(0.0) && s_mid >= C(0.0) ) ? F.h * v_neg : F.h * v_pos;
			
		return F;
	}
	else
	{
		return F_pos;
	}
}

__device__ __forceinline__
FlowVector flux_HLL_y
(
	const FlowVector&           U_neg,
	const FlowVector&           U_pos,
	const SolverParams&     solver_params,
	const SimulationParams& sim_params
)
{
	FlowVector U_neg_rotated = { U_neg.h, U_neg.qy, -U_neg.qx };
	FlowVector U_pos_rotated = { U_pos.h, U_pos.qy, -U_pos.qx };

	FlowVector F_rotated = flux_HLL_x
	(
		U_neg_rotated, 
		U_pos_rotated, 
		solver_params, 
		sim_params
	);

	return { F_rotated.h, -F_rotated.qy, F_rotated.qx };
}