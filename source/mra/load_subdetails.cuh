#pragma once

#include "cuda_runtime.h"

#include "../classes/SubDetail.h"
#include "../classes/SubDetails.h"
#include "../types/HierarchyIndex.h"

__device__ __forceinline__
SubDetailMW load_subdetails_mw
(
	const SubDetails&     d_0,
	const SubDetails&     d_1x,
	const SubDetails&     d_1y,
	const HierarchyIndex& h_idx
)
{
	return
	{
		{
			d_0.alpha[h_idx],
			d_0.beta[h_idx],
			d_0.gamma[h_idx]
		},
		{
			d_1x.alpha[h_idx],
			d_1x.beta[h_idx],
			d_1x.gamma[h_idx]
		},
		{
			d_1y.alpha[h_idx],
			d_1y.beta[h_idx],
			d_1y.gamma[h_idx]
		}
	};
}