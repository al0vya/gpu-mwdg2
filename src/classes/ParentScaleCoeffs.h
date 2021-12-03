#pragma once

#include "real.h"

typedef struct ParentScaleCoeffsHW
{
	real eta;
	real qx;
	real qy;
	real z;

} ParentScaleCoeffsHW;

typedef struct ParentScaleCoeffsMW
{
	ParentScaleCoeffsHW _0;
	ParentScaleCoeffsHW _1x;
	ParentScaleCoeffsHW _1y;

} ParentScaleCoeffsMW;