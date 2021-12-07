#pragma once

#include "FlowVector.h"
#include "LegendreBasis.h"
#include "eval_loc_face_val_dg2.cuh"

typedef struct FlowCoeffs
{
	real h0;
	real h1x;
	real h1y;

	real qx0;
	real qx1x;
	real qx1y;

	real qy0;
	real qy1x;
	real qy1y;

	__device__
	FlowVector local_face_val(const LegendreBasis& basis)
	{
		return
		{
			eval_loc_face_val_dg2(h0,  h1x,  h1y,  basis),
			eval_loc_face_val_dg2(qx0, qx1x, qx1y, basis),
			eval_loc_face_val_dg2(qy0, qy1x, qy1y, basis),
		};
	}
	/*
	__device__
	FlowCoeffs threshold()
	{
		return
		{
			(abs(h0)   > 1e-13) ? h0   : C(0.0),
		    (abs(h1x)  > 1e-13) ? h1x  : C(0.0),
		    (abs(h1y)  > 1e-13) ? h1y  : C(0.0),
		    (abs(qx0)  > 1e-13) ? qx0  : C(0.0),
		    (abs(qx1x) > 1e-13) ? qx1x : C(0.0),
		    (abs(qx1y) > 1e-13) ? qx1y : C(0.0),
		    (abs(qy0)  > 1e-13) ? qy0  : C(0.0),
		    (abs(qy1x) > 1e-13) ? qy1x : C(0.0),
		    (abs(qy1y) > 1e-13) ? qy1y : C(0.0)
		};
	}*/

	__device__
		void set_0
		(
			const FlowVector& v
		)
	{
		h0  = v.h;
		qx0 = v.qx;
		qy0 = v.qy;
	}

	__device__
		void set_1x
		(
			const FlowVector& v
		)
	{
		h1x  = v.h;
		qx1x = v.qx;
		qy1x = v.qy;
	}

	__device__
		void set_1y
		(
			const FlowVector& v
		)
	{
		h1y  = v.h;
		qx1y = v.qx;
		qy1y = v.qy;
	}

	__device__
	inline FlowCoeffs operator=
	(
		const FlowCoeffs& rhs
	)
	{
		h0   = rhs.h0;
		h1x  = rhs.h1x;
		h1y  = rhs.h1y;
		qx0  = rhs.qx0;
		qx1x = rhs.qx1x;
		qx1y = rhs.qx1y;
		qy0  = rhs.qy0;
		qy1x = rhs.qy1x;
		qy1y = rhs.qy1y;

		return *this;
	}
	
	__device__
	inline FlowCoeffs operator+=
	(
		const FlowCoeffs& rhs
	)
	{
		h0   += rhs.h0;
		h1x  += rhs.h1x;
		h1y  += rhs.h1y;
		qx0  += rhs.qx0;
		qx1x += rhs.qx1x;
		qx1y += rhs.qx1y;
		qy0  += rhs.qy0;
		qy1x += rhs.qy1x;
		qy1y += rhs.qy1y;

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
	return
	{
		lhs * rhs.h0,  lhs * rhs.h1x,  lhs * rhs.h1y,
		lhs * rhs.qx0, lhs * rhs.qx1x, lhs * rhs.qx1y,
		lhs * rhs.qy0, lhs * rhs.qy1x, lhs * rhs.qy1y
	};
}

__device__
inline FlowCoeffs operator+
(
	const FlowCoeffs& lhs,
	const FlowCoeffs& rhs
	)
{
	return
	{
		lhs.h0  + rhs.h0,  lhs.h0  + rhs.h1x,  lhs.h0  + rhs.h1y,
		lhs.qx0 + rhs.qx0, lhs.qx0 + rhs.qx1x, lhs.qx0 + rhs.qx1y,
		lhs.qy0 + rhs.qy0, lhs.qy0 + rhs.qy1x, lhs.qy0 + rhs.qy1y
	};
}