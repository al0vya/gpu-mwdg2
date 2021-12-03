#include "read_bound_conds.h"

Depths1D read_bound_conds(const int& test_case)
{
	Depths1D bcs = Depths1D();

	switch (test_case)
	{
	case 1: case 2: // wet c property
		bcs.hl = C(2.0);
		bcs.hr = C(2.0);
		break;
	case 3: // three humps
		bcs.hl = C(2.0);
		bcs.hr = C(0.0);
		break;
	case 4: case 5: // wet dam break
		bcs.hl = C(6.0);
		bcs.hr = C(2.0);
		break;
	case 6:  case 7:  // dry dam break
	case 8:  case 9:  // dry dam break with fric
	case 10: case 11: // overtopping
		bcs.hl = C(6.0);
		bcs.hr = C(0.0);
		break;
	default:
		break;
	}

	return bcs;
}