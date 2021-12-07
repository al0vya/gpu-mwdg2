#pragma once

#include "cuda_runtime.h"

typedef struct DetailChildren
{
	bool detail_0;
	bool detail_1;
	bool detail_2;
	bool detail_3;

	__device__ __forceinline__
	bool has_sig_detail()
	{
		bool is_sig = (detail_0 || detail_1 || detail_2 || detail_3);
		
		return is_sig;
	}

} DetailChildren;