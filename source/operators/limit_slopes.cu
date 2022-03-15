#include "limit_slopes.cuh"

__global__
void limit_slopes
(
    AssembledSolution d_assem_sol,
    Neighbours        d_neighbours,
    SimulationParams  sim_params,
    SolverParams      solver_params,
    real              dx_finest,
    real              dy_finest,
    real              max_h
)
{
	
    int t_idx = threadIdx.x;
    int idx   = blockIdx.x * blockDim.x + t_idx;
    
    if (idx < d_assem_sol.length)
    {
        int level = d_assem_sol.levels[idx];

        if (level == solver_params.L)
        {
            real dx_loc = dx_finest * ( 1 << (solver_params.L - level) );
            real dy_loc = dy_finest * ( 1 << (solver_params.L - level) );

            HierarchyIndex h_idx = d_assem_sol.act_idcs[idx];

            real x = get_x_coord(h_idx, level, solver_params.L, dx_finest);
            real y = get_y_coord(h_idx, level, solver_params.L, dy_finest);

            if ((x >= sim_params.xsz * dx_finest) || (y >= sim_params.ysz * dy_finest)) return;
            
            // ------ //
            // STEP 2 //
            // ------ //
            FlowCoeffs coeffs =
            {
                {
                    d_assem_sol.h0[idx],
                    d_assem_sol.h1x[idx],
                    d_assem_sol.h1y[idx],
                },
                {
                    d_assem_sol.qx0[idx],
                    d_assem_sol.qx1x[idx],
                    d_assem_sol.qx1y[idx],
                },
                {
                    d_assem_sol.qy0[idx],
                    d_assem_sol.qy1x[idx],
                    d_assem_sol.qy1y[idx],
                }
            };

            FlowCoeffs coeffs_n =
            {
                {
                    d_neighbours.north.h0[idx],
                    d_neighbours.north.h1x[idx],
                    d_neighbours.north.h1y[idx],
                },
                {
                    d_neighbours.north.qx0[idx],
                    d_neighbours.north.qx1x[idx],
                    d_neighbours.north.qx1y[idx],
                },
                {
                    d_neighbours.north.qy0[idx],
                    d_neighbours.north.qy1x[idx],
                    d_neighbours.north.qy1y[idx],
                }
            };

            FlowCoeffs coeffs_e =
            {
                {
                    d_neighbours.east.h0[idx],
                    d_neighbours.east.h1x[idx],
                    d_neighbours.east.h1y[idx],
                },
                {
                    d_neighbours.east.qx0[idx],
                    d_neighbours.east.qx1x[idx],
                    d_neighbours.east.qx1y[idx],
                },
                {
                    d_neighbours.east.qy0[idx],
                    d_neighbours.east.qy1x[idx],
                    d_neighbours.east.qy1y[idx],
                }
            };

            FlowCoeffs coeffs_s =
            {
                {
                    d_neighbours.south.h0[idx],
                    d_neighbours.south.h1x[idx],
                    d_neighbours.south.h1y[idx],
                },
                {
                    d_neighbours.south.qx0[idx],
                    d_neighbours.south.qx1x[idx],
                    d_neighbours.south.qx1y[idx],
                },
                {
                    d_neighbours.south.qy0[idx],
                    d_neighbours.south.qy1x[idx],
                    d_neighbours.south.qy1y[idx],
                }
            };

            FlowCoeffs coeffs_w =
            {
                {
                    d_neighbours.west.h0[idx],
                    d_neighbours.west.h1x[idx],
                    d_neighbours.west.h1y[idx],
                },
                {
                    d_neighbours.west.qx0[idx],
                    d_neighbours.west.qx1x[idx],
                    d_neighbours.west.qx1y[idx],
                },
                {
                    d_neighbours.west.qy0[idx],
                    d_neighbours.west.qy1x[idx],
                    d_neighbours.west.qy1y[idx],
                }
            };

            const PlanarCoefficients z   = { d_assem_sol.z0[idx],   d_assem_sol.z1x[idx],   d_assem_sol.z1y[idx] };
            const PlanarCoefficients z_n = { d_neighbours.north.z0[idx], d_neighbours.north.z1x[idx], d_neighbours.north.z1y[idx] };
            const PlanarCoefficients z_e = { d_neighbours.east.z0[idx],  d_neighbours.east.z1x[idx],  d_neighbours.east.z1y[idx] };
            const PlanarCoefficients z_s = { d_neighbours.south.z0[idx], d_neighbours.south.z1x[idx], d_neighbours.south.z1y[idx] };
            const PlanarCoefficients z_w = { d_neighbours.west.z0[idx],  d_neighbours.west.z1x[idx],  d_neighbours.west.z1y[idx] };

            const PlanarCoefficients eta   = z   + coeffs.h;
            const PlanarCoefficients eta_n = z_n + coeffs_n.h;
            const PlanarCoefficients eta_e = z_e + coeffs_e.h;
            const PlanarCoefficients eta_s = z_s + coeffs_s.h;
            const PlanarCoefficients eta_w = z_w + coeffs_w.h;

            int level_n = d_neighbours.north.levels[idx];
            int level_e = d_neighbours.east.levels[idx];
            int level_s = d_neighbours.south.levels[idx];
            int level_w = d_neighbours.west.levels[idx];

            HierarchyIndex h_idx_n = d_neighbours.north.act_idcs[idx];
            HierarchyIndex h_idx_e = d_neighbours.east.act_idcs[idx];
            HierarchyIndex h_idx_s = d_neighbours.south.act_idcs[idx];
            HierarchyIndex h_idx_w = d_neighbours.west.act_idcs[idx];

            // between local && east,  x, y unit is (1.0, 0.5)
            // between local && west,  x, y unit is (0.0, 0.5)
            // between local && north, x, y unit is (0.5, 1.0)
            // between local && south, x, y unit is (0.5, 0.0)

            // ------- //
            // STEP 3a //
            // ------- //
            LegendreBasis basis_n = get_leg_basis(h_idx, h_idx_n, level_n, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, NORTH);
            LegendreBasis basis_e = get_leg_basis(h_idx, h_idx_e, level_e, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, EAST);
            LegendreBasis basis_s = get_leg_basis(h_idx, h_idx_s, level_s, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, SOUTH);
            LegendreBasis basis_w = get_leg_basis(h_idx, h_idx_w, level_w, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, WEST);
            
            real tol_Krivo = C(1.0);

            Slopes eta_limited = get_limited_slopes
            (
                eta,
                eta_n,
                eta_e,
                eta_s,
                eta_w,
                basis_n,
                basis_e,
                basis_s,
                basis_w,
                dx_finest,
                dy_finest,
                tol_Krivo
            );

            Slopes qx_limited = get_limited_slopes
            (
                coeffs.qx,
                coeffs_n.qx,
                coeffs_e.qx,
                coeffs_s.qx,
                coeffs_w.qx,
                basis_n,
                basis_e,
                basis_s,
                basis_w,
                dx_finest,
                dy_finest,
                tol_Krivo
            );

            Slopes qy_limited = get_limited_slopes
            (
                coeffs.qy,
                coeffs_n.qy,
                coeffs_e.qy,
                coeffs_s.qy,
                coeffs_w.qy,
                basis_n,
                basis_e,
                basis_s,
                basis_w,
                dx_finest,
                dy_finest,
                tol_Krivo
            );

            const bool above_h_min_limiter_x = ( C(0.01) * max_h < min( coeffs.h._0, min(coeffs_e.h._0, coeffs_w.h._0) ) );
            const bool above_h_min_limiter_y = ( C(0.01) * max_h < min( coeffs.h._0, min(coeffs_n.h._0, coeffs_s.h._0) ) );

            real& h1x = d_assem_sol.h1x[idx];
            real& h1y = d_assem_sol.h1y[idx];
            
            real& qx1x = d_assem_sol.qx1x[idx];
            real& qx1y = d_assem_sol.qx1y[idx];
            
            real& qy1x = d_assem_sol.qy1x[idx];
            real& qy1y = d_assem_sol.qy1y[idx];

            h1x = (above_h_min_limiter_x) ? eta_limited._1x - z._1x : h1x;
            h1y = (above_h_min_limiter_y) ? eta_limited._1y - z._1y : h1y;
            
            qx1x = (above_h_min_limiter_x) ? qx_limited._1x : qx1x;
            qx1y = (above_h_min_limiter_y) ? qx_limited._1y : qx1y;
            
            qy1x = (above_h_min_limiter_x) ? qy_limited._1x : qy1x;
            qy1y = (above_h_min_limiter_y) ? qy_limited._1y : qy1y;
        }
    }
}