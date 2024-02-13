#pragma once

#include "../types/real.h"

typedef struct Depths1D
{
	real hl;
	real hr;

	Depths1D(const int& test_case)
	{
		switch (test_case)
		{
		case 1: case 2: // wet c property
			this->hl = C(6.0);
			this->hr = C(6.0);
			break;
		case 3: case 4: // wet/dry c property
			this->hl = C(2.0);
			this->hr = C(2.0);
			break;
		case 5: case 6:   // wet dam break
		case 11: case 12: // wet overtopping
			this->hl = C(6.0);
			this->hr = C(2.0);
			break;
		case 7:  case 8:  // dry dam break
		case 9:  case 10: // dry dam break with fric
		case 13: case 14: // dry overtopping
			this->hl = C(6.0);
			this->hr = C(0.0);
			break;
		case 17: // three humps
			this->hl = C(2.0);
			this->hr = C(0.0);
			break;
		default:
			break;
		}
	}

} Depths1D;

/*
example of a designated initialiser
requires /std:c++latest
the (slight) advantage is bcs will be initialised upon construction
there is never a time when bcs is (partially) uninitialised

Depths1D bcs = {
	.hl = C(2.0),
	.hr = C(0.0),
	.ql = C(0.0),
	.qr = C(0.0),
	.reflectUp = C(0.0),
	.reflectDown = C(0.0),
	.hImposedUp = C(0.0),
	.qxImposedUp = C(0.0),
	.hImposedDown = C(0.0),
	.qxImposedDown = C(0.0)
};
*/