#pragma once

#include "FlowVector.h"

#include "LegendreBasis.h"

__device__ __forceinline__
real eval_loc_face_val_dg2
(
	const real&          s0,
	const real&          s1x,
	const real&          s1y,
	const LegendreBasis& basis
)
{
	return { s0 * basis._0 + s1x * basis._1x + s1y * basis._1y };
}