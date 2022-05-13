#pragma once

#include "cuda_runtime.h"

#include "Detail.h"
#include "Details.h"
#include "HierarchyIndex.h"

__device__
DetailHW load_details_hw
(
	const Details&  d_details,
	const HierarchyIndex& h_idx
);

__device__
DetailMW load_details_mw
(
	const Details&  d_details,
	const HierarchyIndex& h_idx
);