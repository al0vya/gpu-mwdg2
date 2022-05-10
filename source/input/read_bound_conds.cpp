#include "read_bound_conds.h"

Depths1D read_bound_conds(const int& test_case)
{
	Depths1D bcs = Depths1D();

	switch (test_case)
	{
	case 1: case 2: // wet c property
		bcs.hl = C(6.0);
		bcs.hr = C(6.0);
		break;
	case 3: case 4: // wet/dry c property
		bcs.hl = C(2.0);
		bcs.hr = C(2.0);
		break;
	case 5: case 6:   // wet dam break
	case 11: case 12: // wet overtopping
		bcs.hl = C(6.0);
		bcs.hr = C(2.0);
		break;
	case 7:  case 8:  // dry dam break
	case 9:  case 10: // dry dam break with fric
	case 13: case 14: // dry overtopping
		bcs.hl = C(6.0);
		bcs.hr = C(0.0);
		break;
	case 17: // three humps
		bcs.hl = C(2.0);
		bcs.hr = C(0.0);
		break;
	default:
		break;
	}

	return bcs;
}