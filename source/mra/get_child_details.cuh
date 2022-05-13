#pragma once

#include "cuda_runtime.h"

#include "DetailChildren.h"

__device__
DetailChildren get_child_details
(
	bool* shared_child_details,
	int   t_idx
);