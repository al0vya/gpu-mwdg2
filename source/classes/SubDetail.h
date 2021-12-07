#pragma once

#include "cuda_runtime.h"

#include <math.h>

#include "real.h"


typedef struct SubDetailHW
{
	real alpha;
	real beta;
	real gamma;

	__device__  __forceinline__ real get_max()
	{
		real max_detail = C(0.0);

		max_detail = max( abs(alpha), abs(beta) );
		max_detail = max( abs(gamma), max_detail);

		return max_detail;
	}
	
} SubDetailHW;

typedef struct SubDetailMW
{
	SubDetailHW _0;
	SubDetailHW _1x;
	SubDetailHW _1y;

	__device__  __forceinline__ real get_max()
	{
		real max_detail = C(0.0);

		max_detail = max( _0.get_max(), _1x.get_max() );
		max_detail = max(_1y.get_max(), max_detail);

		return max_detail;
	}

} SubDetailMW;