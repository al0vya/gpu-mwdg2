#pragma once

#include <cstdio>
#include <algorithm>

#include "are_reals_equal.h"

real compare_array_with_file_real
(
	const char* dirroot,
	const char* filename,
	real*       h_array,
	const int&  array_length
);