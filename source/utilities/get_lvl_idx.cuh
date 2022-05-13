#pragma once

#include "cuda_runtime.h"

#include "HierarchyIndex.h"

__host__ __device__
HierarchyIndex get_lvl_idx(int level);