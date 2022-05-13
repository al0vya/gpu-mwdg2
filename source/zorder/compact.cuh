#pragma once

#include "cuda_runtime.h"

#include "Coordinate.h"
#include "MortonCode.h"

// remove the even bits and squash together the odd bits of a Morton code
__host__ __device__
Coordinate compact(MortonCode code);