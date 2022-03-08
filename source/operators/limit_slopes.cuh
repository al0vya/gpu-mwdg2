#pragma once

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

            const real h_n_neg = coeffs.h0 + sqrt( C(3.0) ) * coeffs.h1y;
            const real h_e_neg = coeffs.h0 + sqrt( C(3.0) ) * coeffs.h1x;
            const real h_s_pos = coeffs.h0 - sqrt( C(3.0) ) * coeffs.h1y;
            const real h_w_pos = coeffs.h0 - sqrt( C(3.0) ) * coeffs.h1x;
            
            const real h_n_pos = coeffs_n.h0 - sqrt( C(3.0) ) * coeffs_n.h1y;
            const real h_e_pos = coeffs_e.h0 - sqrt( C(3.0) ) * coeffs_e.h1x;
            const real h_s_neg = coeffs_s.h0 + sqrt( C(3.0) ) * coeffs_s.h1y;
            const real h_w_neg = coeffs_w.h0 + sqrt( C(3.0) ) * coeffs_w.h1x;
            
            const real z_n_neg = z0 + sqrt( C(3.0) ) * z1y;
            const real z_e_neg = z0 + sqrt( C(3.0) ) * z1x;
            const real z_s_pos = z0 - sqrt( C(3.0) ) * z1y;
            const real z_w_pos = z0 - sqrt( C(3.0) ) * z1x;
            
            const real z_n_pos = z0_w - sqrt( C(3.0) ) * z1y_w;
            const real z_e_pos = z0_e - sqrt( C(3.0) ) * z1x_e;
            const real z_s_neg = z0_s + sqrt( C(3.0) ) * z1y_s;
            const real z_w_neg = z0_n + sqrt( C(3.0) ) * z1x_n;
            
            const real eta_n_neg = h_n_neg + z_n_neg;
            const real eta_e_neg = h_e_neg + z_e_neg;
            const real eta_s_pos = h_s_pos + z_s_pos;
            const real eta_w_pos = h_w_pos + z_w_pos;

            const real eta_n_pos = h_n_pos + z_n_pos;
            const real eta_e_pos = h_e_pos + z_e_pos;
            const real eta_s_neg = h_s_neg + z_s_neg;
            const real eta_w_neg = h_w_neg + z_w_neg;

            const real eta_jump_n = abs(eta_n_pos - eta_n_neg);
            const real eta_jump_e = abs(eta_e_pos - eta_e_neg);
            const real eta_jump_s = abs(eta_s_pos - eta_s_neg);
            const real eta_jump_w = abs(eta_w_pos - eta_w_neg);

            const real denom_1_x = C(0.5) * dx_finest;
            const real denom_1_y = C(0.5) * dy_finest;

            const real denom_2_x;
        }
    }
}