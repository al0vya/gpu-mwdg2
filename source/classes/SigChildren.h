#pragma once

#include "cuda_runtime.h"

typedef struct SigChildren
{
	bool child_0;
	bool child_1;
	bool child_2;
	bool child_3;

	__device__
	SigChildren(char4 children)
	:
		child_0(children.x),
		child_1(children.y),
		child_2(children.z),
		child_3(children.w)
	{}

	__device__ __forceinline__
	bool has_sig_detail()
	{
		return (child_0 || child_1 || child_2 || child_3);
	}

} SigChildren;