#pragma once

#include "../types/Coordinate.h"
#include "../types/real.h"
#include "../input/read_keyword_str.h"
#include "../classes/SolverParams.h"

class SimulationParams
{
public:
    SimulationParams();

    SimulationParams
    (
        const int&  test_case,
        const char* input_filename,
        const int&  max_ref_lvl
    );

	real       xmin      = C(0.0);
	real       xmax      = C(0.0);
	real       ymin      = C(0.0);
	real       ymax      = C(0.0);
	Coordinate xsz       = 0;
	Coordinate ysz       = 0;
	real       g         = C(9.80665);
	real       time      = C(0.0);
	real       manning   = C(0.0);
	bool       is_monai  = false;
	bool       is_oregon = false;
	int        test_case = -1;
};