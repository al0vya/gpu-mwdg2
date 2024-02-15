#pragma once

#include <cmath>

#include "../types/real.h"

bool are_reals_equal
(
	const real& a,
	const real& b,
	const real& epsilon = C(1e-6)
);