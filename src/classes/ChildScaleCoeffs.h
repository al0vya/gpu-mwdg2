#pragma once

#include "ScaleChildren.h"

typedef struct ChildScaleCoeffsHW
{
	ScaleChildrenHW eta;
	ScaleChildrenHW qx;
	ScaleChildrenHW qy;
	ScaleChildrenHW z;

} ChildScaleCoeffsHW;

typedef struct ChildScaleCoeffsMW
{
	ScaleChildrenMW eta;
	ScaleChildrenMW qx;
	ScaleChildrenMW qy;
	ScaleChildrenMW z;

} ChildScaleCoeffsMW;