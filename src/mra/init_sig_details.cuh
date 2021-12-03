#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void init_sig_details
(
	bool* d_sig_details, 
	int   num_details
);