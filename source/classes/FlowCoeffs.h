#pragma once

#include "FlowVector.h"
#include "PlanarCoefficients.h"
#include "LegendreBasis.h"
#include "eval_loc_face_val_dg2.cuh"

typedef struct FlowCoeffs
{
	PlanarCoefficients h;
	PlanarCoefficients qx;
	PlanarCoefficients qy;

	__device__
	FlowVector local_face_val(const LegendreBasis& basis)
	{
		return
		{
			eval_loc_face_val_dg2(h,  basis),
			eval_loc_face_val_dg2(qx, basis),
			eval_loc_face_val_dg2(qy, basis)
		};
	}
	
	__device__
	void set_0
	(
		const FlowVector& v
	)
	{
		h._0  = v.h;
		qx._0 = v.qx;
		qy._0 = v.qy;
	}

	__device__
	void set_1x
	(
		const FlowVector& v
	)
	{
		h._1x  = v.h;
		qx._1x = v.qx;
		qy._1x = v.qy;
	}

	__device__
	void set_1y
	(
		const FlowVector& v
	)
	{
		h._1y  = v.h;
		qx._1y = v.qx;
		qy._1y = v.qy;
	}

	__device__
	inline FlowCoeffs operator=
	(
		const FlowCoeffs& rhs
	)
	{
		h  = rhs.h;
		qx = rhs.qx;
		qy = rhs.qy;
		
		return *this;
	}
	
	__device__
	inline FlowCoeffs operator+=
	(
		const FlowCoeffs& rhs
	)
	{
		h  += rhs.h;
		qx += rhs.qx;
		qy += rhs.qy;

		return *this;
	}

} FlowCoeffs;

__device__
inline FlowCoeffs operator*
(
	const real& lhs,
	const FlowCoeffs& rhs
)
{
	return { lhs * rhs.h, lhs * rhs.qx, lhs * rhs.qy };
}

__device__
inline FlowCoeffs operator+
(
	const FlowCoeffs& lhs,
	const FlowCoeffs& rhs
)
{
	return { lhs.h + rhs.h, lhs.qx + rhs.qx,  lhs.qy + rhs.qy };
}