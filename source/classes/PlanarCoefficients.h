#pragma once

#include "cuda_runtime.h"

#include "real.h"

typedef struct PlanarCoefficients
{
	real _0;
	real _1x;
	real _1y;

	__device__
	inline PlanarCoefficients operator=
	(
		const PlanarCoefficients& rhs
	)
	{
		_0  = rhs._0;
		_1x = rhs._1x;
		_1y = rhs._1y;

		return *this;
	}

	__device__
	inline PlanarCoefficients operator+=
	(
		const PlanarCoefficients& rhs
	)
	{
		_0  += rhs._0;
		_1x += rhs._1x;
		_1y += rhs._1y;

		return *this;
	}

} PlanarCoefficients;

__device__
inline PlanarCoefficients operator*
(
	const real& lhs,
	const PlanarCoefficients& rhs
)
{
	return { lhs * rhs._0, lhs * rhs._1x, lhs * rhs._1y };
}

__device__
inline PlanarCoefficients operator+
(
	const PlanarCoefficients& lhs,
	const PlanarCoefficients& rhs
)
{
	return { lhs._0 + rhs._0,  lhs._1x + rhs._1x,  lhs._1y + rhs._1y };
}