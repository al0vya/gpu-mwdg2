#pragma once

#include "FlowVector.h"
#include "PlanarCoefficients.h"
#include "LegendreBasis.h"

__host__ __device__
real eval_loc_face_val_dg2
(
	const PlanarCoefficients& s,
	const LegendreBasis&      basis
);