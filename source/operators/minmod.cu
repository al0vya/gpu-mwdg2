#include "minmod.cuh"

__device__
real minmod
(
    const real& a,
    const real& b,
    const real& c
)
{
    if ( ( a * b > C(0.0) ) && ( a * c > C(0.0) ) )
    {
        const real sign_a = ( a > C(0.0) ) - ( a < C(0.0) );

        return min( abs(a), min( abs(b), abs(c) ) ) * sign_a;
    }
    else
    {
        return C(0.0);
    }
}