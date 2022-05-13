#pragma once

#include "ScaleChildren.h"

typedef struct ChildScaleCoefficients
{
	ScaleChildrenHW eta;
	ScaleChildrenHW qx;
	ScaleChildrenHW qy;
	ScaleChildrenHW z;

} ChildScaleCoefficients;