#include "are_reals_equal.h"

bool are_reals_equal
(
	const real& a,
	const real& b,
	const real& epsilon
)
{
	return fabs(a - b) <= epsilon;
}