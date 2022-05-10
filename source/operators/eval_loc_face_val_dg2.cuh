#pragma once

#include "FlowVector.h"
#include "PlanarCoefficients.h"
#include "LegendreBasis.h"

__host__ __device__ __forceinline__
real eval_loc_face_val_dg2
(
	const PlanarCoefficients& s,
	const LegendreBasis&      basis
)
{
	return s._0 * basis._0 + s._1x * basis._1x + s._1y * basis._1y;
}