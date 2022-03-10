#pragma once

#include "minmod.cuh"

__global__
void limit_slopes
(

)
{
	
    int t_idx = threadIdx.x;
    int idx   = blockIdx.x * blockDim.x + t_idx;
    
    if (idx < d_assem_sol.length)
    {
        int level = d_assem_sol.levels[idx];

        if (level == solver_params.L)
        {
            FlowCoeffs coeffs =
            {
                d_assem_sol.h0[idx],
                d_assem_sol.h1x[idx],
                d_assem_sol.h1y[idx],

                d_assem_sol.qx0[idx],
                d_assem_sol.qx1x[idx],
                d_assem_sol.qx1y[idx],

                d_assem_sol.qy0[idx],
                d_assem_sol.qy1x[idx],
                d_assem_sol.qy1y[idx]
            };

            FlowCoeffs coeffs_n
            {
                d_neighbours.north.h0[idx],
                d_neighbours.north.h1x[idx],
                d_neighbours.north.h1y[idx],

                d_neighbours.north.qx0[idx],
                d_neighbours.north.qx1x[idx],
                d_neighbours.north.qx1y[idx],

                d_neighbours.north.qy0[idx],
                d_neighbours.north.qy1x[idx],
                d_neighbours.north.qy1y[idx]
            };

            FlowCoeffs coeffs_e
            {
                d_neighbours.east.h0[idx],
                d_neighbours.east.h1x[idx],
                d_neighbours.east.h1y[idx],

                d_neighbours.east.qx0[idx],
                d_neighbours.east.qx1x[idx],
                d_neighbours.east.qx1y[idx],

                d_neighbours.east.qy0[idx],
                d_neighbours.east.qy1x[idx],
                d_neighbours.east.qy1y[idx]
            };

            FlowCoeffs coeffs_s
            {
                d_neighbours.south.h0[idx],
                d_neighbours.south.h1x[idx],
                d_neighbours.south.h1y[idx],

                d_neighbours.south.qx0[idx],
                d_neighbours.south.qx1x[idx],
                d_neighbours.south.qx1y[idx],

                d_neighbours.south.qy0[idx],
                d_neighbours.south.qy1x[idx],
                d_neighbours.south.qy1y[idx]
            };

            FlowCoeffs coeffs_w
            {
                d_neighbours.west.h0[idx],
                d_neighbours.west.h1x[idx],
                d_neighbours.west.h1y[idx],

                d_neighbours.west.qx0[idx],
                d_neighbours.west.qx1x[idx],
                d_neighbours.west.qx1y[idx],

                d_neighbours.west.qy0[idx],
                d_neighbours.west.qy1x[idx],
                d_neighbours.west.qy1y[idx]
            };

            const real z0  = d_assem_sol.z0[idx];
            const real z1x = d_assem_sol.z1x[idx];
            const real z1y = d_assem_sol.z1y[idx];
            
            const real z0_n  = d_neighbours.north.z0[idx];
            const real z1x_n = d_neighbours.north.z1x[idx];
            const real z1y_n = d_neighbours.north.z1y[idx];
            
            const real z0_e  = d_neighbours.east.z0[idx];
            const real z1x_e = d_neighbours.east.z1x[idx];
            const real z1y_e = d_neighbours.east.z1y[idx];
            
            const real z0_s  = d_neighbours.south.z0[idx];
            const real z1x_s = d_neighbours.south.z1x[idx];
            const real z1y_s = d_neighbours.south.z1y[idx];
            
            const real z0_w  = d_neighbours.west.z0[idx];
            const real z1x_w = d_neighbours.west.z1x[idx];
            const real z1y_w = d_neighbours.west.z1y[idx];

            const real eta_n_neg = (coeffs.h0 + z0) + sqrt( C(3.0) ) * (coeffs.h1y + z1y);
            const real eta_e_neg = (coeffs.h0 + z0) + sqrt( C(3.0) ) * (coeffs.h1x + z1x);
            const real eta_s_pos = (coeffs.h0 + z0) - sqrt( C(3.0) ) * (coeffs.h1y + z1y);
            const real eta_w_pos = (coeffs.h0 + z0) - sqrt( C(3.0) ) * (coeffs.h1x + z1x);

            const real eta_n_pos = (coeffs_n.h0 + z0_n) - sqrt( C(3.0) ) * (coeffs_n.h1y + z1y_n);
            const real eta_e_pos = (coeffs_e.h0 + z0_e) - sqrt( C(3.0) ) * (coeffs_e.h1x + z1x_e);
            const real eta_s_neg = (coeffs_s.h0 + z0_s) + sqrt( C(3.0) ) * (coeffs_s.h1y + z1y_s);
            const real eta_w_neg = (coeffs_w.h0 + z0_w) + sqrt( C(3.0) ) * (coeffs_w.h1x + z1x_w);
            
            const real eta_jump_n = abs(eta_n_pos - eta_n_neg);
            const real eta_jump_e = abs(eta_e_pos - eta_e_neg);
            const real eta_jump_s = abs(eta_s_pos - eta_s_neg);
            const real eta_jump_w = abs(eta_w_pos - eta_w_neg);

            const real eta_norm_x = max( abs( coeffs.h0 + z0 - (coeffs_w.h0 + z0_w) ), abs( coeffs_e.h0 + z0_e - (coeffs.h0 + z0) );
            const real eta_norm_y = max( abs( coeffs.h0 + z0 - (coeffs_s.h0 + z0_s) ), abs( coeffs_n.h0 + z0_n - (coeffs.h0 + z0) );

            const real eta_DS_e = ( eta_norm_x > C(1e-12) ) ? eta_jump_e / (C(0.5) * dx_finest * eta_norm_x) : C(0.0);
            const real eta_DS_w = ( eta_norm_x > C(1e-12) ) ? eta_jump_w / (C(0.5) * dx_finest * eta_norm_x) : C(0.0);

            const real eta_DS_n = ( eta_norm_y > C(1e-12) ) ? eta_jump_n / (C(0.5) * dy_finest * eta_norm_y) : C(0.0);
            const real eta_DS_s = ( eta_norm_y > C(1e-12) ) ? eta_jump_s / (C(0.5) * dy_finest * eta_norm_y) : C(0.0);

            const real eta1x_limit = minmod
            (
                coeffs.h1x + z1x,
                ( coeffs.h0 + z0 - (coeffs_w.h0 + z0_w) ) / sqrt( C(3.0) ),
                ( coeffs_e.h0 + z0_e - (coeffs.h0 + z0) ) / sqrt( C(3.0) )
            );
        }
    }
}