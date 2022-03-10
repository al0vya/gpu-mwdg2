#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__ __forceinline__
real minmod
(
    const real& a,
    const real& b,
    const real& c
)
{
    if ( ( a * b > C(0.0) ) && ( a * c > C(0.0) ) )
    {
        const int sign_a = ( a > C(0.0) ) - ( a < C(0.0) );

        return min( abs(a), min( abs(b), abs(c) ) ) * sign_a;
    }
    else
    {
        return C(0.0);
    }
}