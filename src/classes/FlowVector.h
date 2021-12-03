#pragma once

#include "cuda_runtime.h"

#include "real.h"

typedef struct FlowVector
{
    real h;
    real qx;
    real qy;

    __device__ __forceinline__
    real get_speed
    (
        const real& q,
        const real& tol_h
    )
    {
        return (h < tol_h) ? 0 : q / h;
    }

    __device__ __forceinline__
    real calc_h_star
    (
        const real& z,
        const real& z_intermediate
    )
    {
        real eta = h + z;

        return max(C(0.0), eta - z_intermediate);
    }

    __device__ __forceinline__
    FlowVector get_star
    (
        const real& z,
        const real& z_intermediate,
        const real& tol_h
    )
    {
        real h_star  = calc_h_star(z, z_intermediate);
        real qx_star = h_star * get_speed(qx, tol_h);
        real qy_star = h_star * get_speed(qy, tol_h);

        return { h_star, qx_star, qy_star };
    }

    __device__ __forceinline__
    FlowVector phys_flux_x
    (
        const real& tol_h,
        const real& g
    )
    {
        if (h < tol_h)
        {
            return { 0, 0, 0 };
        }
        else
        {
            return { qx, qx * qx / h + g * h * h / C(2.0), qx * qy / h };
        }
    }

    __device__ __forceinline__
    FlowVector phys_flux_y
    (
        const real& tol_h,
        const real& g
    )
    {
        if (h < tol_h)
        {
            return { 0, 0, 0 };
        }
        else
        {
            return { qy, qx * qy / h, qy * qy / h + g * h * h / C(2.0) };
        }
    }

} FlowVector;

__host__ __device__
inline FlowVector operator+
(
    const FlowVector& lhs,
    const FlowVector& rhs
)
{
    return { lhs.h + rhs.h, lhs.qx + rhs.qx, lhs.qy + rhs.qy };
}

__host__ __device__
inline FlowVector operator-
(
    const FlowVector& lhs,
    const FlowVector& rhs
)
{
    return { lhs.h - rhs.h, lhs.qx - rhs.qx, lhs.qy - rhs.qy };
}

__host__ __device__
inline FlowVector operator*
(
    const real& lhs,
    const FlowVector& rhs
)
{
    return { lhs * rhs.h, lhs * rhs.qx, lhs * rhs.qy };
}

__host__ __device__
inline FlowVector operator/
(
    const FlowVector& lhs,
    const real& rhs
)
{
    return { lhs.h / rhs, lhs.qx / rhs, lhs.qy / rhs };
}