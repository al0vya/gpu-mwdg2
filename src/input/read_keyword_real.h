#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "real.h"

real read_keyword_real
(
	const char* filename,
	const char* keyword,
	const int&  num_char
);