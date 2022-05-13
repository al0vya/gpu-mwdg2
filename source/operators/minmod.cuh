#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__
real minmod
(
    const real& a,
    const real& b,
    const real& c
);