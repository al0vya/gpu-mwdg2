#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "ScaleChildren.h"
#include "Filters.h"

__device__
real encode_scale(const ScaleChildrenHW& u);

__device__
real encode_scale_0(const ScaleChildrenMW& u);

__device__
real encode_scale_1x(const ScaleChildrenMW& u);

__device__
real encode_scale_1y(const ScaleChildrenMW& u);