#pragma once

#include "../types/real.h"

typedef struct ScaleChildrenHW
{
	real child_0;
	real child_1;
	real child_2;
	real child_3;

} ScaleChildrenHW;

typedef struct ScaleChildrenMW
{
	ScaleChildrenHW _0;
	ScaleChildrenHW _1x;
	ScaleChildrenHW _1y;

} ScaleChildrenMW;