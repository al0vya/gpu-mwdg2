#pragma once

#include "cuda_runtime.h"

#include "Detail.h"
#include "Details.h"

__device__
void store_details
(
	const DetailHW& detail, 
	const Details&  d_details, 
	const int&      h_idx
);

__device__
void store_details
(
	const DetailMW& detail, 
	const Details&  d_details, 
	const int&      h_idx
);