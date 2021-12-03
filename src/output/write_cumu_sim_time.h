#pragma once

#include "BLOCK_VAR_MACROS.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "real.h"

void write_cumu_sim_time
(
	const clock_t              start,
	const real                 time_now,
	const char*                respath,
	const bool                 first_t_step
);