#include "dg2_update.cuh"

__global__
void dg2_update
(
    Neighbours        d_neighbours,
    AssembledSolution d_assem_sol_load,
    AssembledSolution d_assem_sol_store,
    SolverParams      solver_params,
    SimulationParams  sim_params,
    real              dx_finest,
    real              dy_finest,
    real              dt,
    int               test_case,
    real*             d_dt_CFL,
    bool              rkdg2
)
{
    typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

    __shared__ union
    {
        typename block_scan::TempStorage temp_storage;
        HierarchyIndex indices[THREADS_PER_BLOCK];

    } shared;

    int t_idx = threadIdx.x;
    int idx   = blockIdx.x * blockDim.x + t_idx;
    
    int is_wet = 0;

    int thread_prefix_sum = 0;

    int num_wet = 0;

    if (idx < d_assem_sol_load.length)
    {
        real z   = d_assem_sol_load.z0[idx];
        real h   = d_assem_sol_load.h0[idx];
        real h_n = d_neighbours.north.h0[idx];
        real h_e = d_neighbours.east.h0[idx];
        real h_s = d_neighbours.south.h0[idx];
        real h_w = d_neighbours.west.h0[idx];

        bool highwall = ( ( fabs( z - solver_params.wall_height ) < C(1e-10) ) && (test_case == 0) );

        is_wet =
        (
            (
                h   >= solver_params.tol_h ||
                h_n >= solver_params.tol_h ||
                h_e >= solver_params.tol_h ||
                h_s >= solver_params.tol_h ||
                h_w >= solver_params.tol_h
            )
            &&
            !highwall
        );

        d_dt_CFL[idx] = solver_params.min_dt;
    }

    block_scan(shared.temp_storage).ExclusiveSum
    (
        is_wet,
        thread_prefix_sum,
        num_wet
    );

    __syncthreads();

    if (is_wet) shared.indices[thread_prefix_sum] = idx;

    __syncthreads();

    if (t_idx >= num_wet) return;

    idx = shared.indices[t_idx];

    int level = d_assem_sol_load.levels[idx];
    
    real dx_loc = dx_finest * ( 1 << (solver_params.L - level) );
    real dy_loc = dy_finest * ( 1 << (solver_params.L - level) );
    
    HierarchyIndex h_idx   = d_assem_sol_load.act_idcs[idx];
    
    real x = get_x_coord(h_idx, level, solver_params.L, dx_finest);
    real y = get_y_coord(h_idx, level, solver_params.L, dy_finest);
    
    if ( (x >= sim_params.xsz * dx_finest) || (y >= sim_params.ysz * dy_finest) ) return;
    
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
    
    LegendreBasis basis_n = get_leg_basis(h_idx, h_idx_n, level_n, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, NORTH);
    LegendreBasis basis_e = get_leg_basis(h_idx, h_idx_e, level_e, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, EAST);
    LegendreBasis basis_s = get_leg_basis(h_idx, h_idx_s, level_s, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, SOUTH);
    LegendreBasis basis_w = get_leg_basis(h_idx, h_idx_w, level_w, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, WEST);
    
    LegendreBasis basis_n_loc = { C(1.0), C(0.0), sqrt( C(3.0) ) } ;
    LegendreBasis basis_e_loc = { C(1.0), sqrt( C(3.0) ), C(0.0) } ;
    LegendreBasis basis_s_loc = { C(1.0), C(0.0), -sqrt( C(3.0) ) };
    LegendreBasis basis_w_loc = { C(1.0), -sqrt( C(3.0) ), C(0.0) };
    
    FlowCoeffs coeffs =
    {
        {
            d_assem_sol_load.h0[idx],
            d_assem_sol_load.h1x[idx],
            d_assem_sol_load.h1y[idx],
        },
        {
            d_assem_sol_load.qx0[idx],
            d_assem_sol_load.qx1x[idx],
            d_assem_sol_load.qx1y[idx],
        },
        {
            d_assem_sol_load.qy0[idx],
            d_assem_sol_load.qy1x[idx],
            d_assem_sol_load.qy1y[idx],
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

    PlanarCoefficients z_planar   = { d_assem_sol_load.z0[idx],   d_assem_sol_load.z1x[idx],   d_assem_sol_load.z1y[idx] };
    PlanarCoefficients z_planar_n = { d_neighbours.north.z0[idx], d_neighbours.north.z1x[idx], d_neighbours.north.z1y[idx] };
    PlanarCoefficients z_planar_e = { d_neighbours.east.z0[idx],  d_neighbours.east.z1x[idx],  d_neighbours.east.z1y[idx] };
    PlanarCoefficients z_planar_s = { d_neighbours.south.z0[idx], d_neighbours.south.z1x[idx], d_neighbours.south.z1y[idx] };
    PlanarCoefficients z_planar_w = { d_neighbours.west.z0[idx],  d_neighbours.west.z1x[idx],  d_neighbours.west.z1y[idx] };

    // LFVs from neighbour cells
    real z_n_pos = eval_loc_face_val_dg2(z_planar_n, basis_n);
    real z_e_pos = eval_loc_face_val_dg2(z_planar_e, basis_e);
    real z_s_neg = eval_loc_face_val_dg2(z_planar_s, basis_s);
    real z_w_neg = eval_loc_face_val_dg2(z_planar_w, basis_w);

    // LFVs of local cell
    real z_n_neg = eval_loc_face_val_dg2(z_planar, basis_n_loc);
    real z_e_neg = eval_loc_face_val_dg2(z_planar, basis_e_loc);
    real z_s_pos = eval_loc_face_val_dg2(z_planar, basis_s_loc);
    real z_w_pos = eval_loc_face_val_dg2(z_planar, basis_w_loc);

    real z_inter_n = max(z_n_neg, z_n_pos);
    real z_inter_e = max(z_e_neg, z_e_pos);
    real z_inter_s = max(z_s_neg, z_s_pos);
    real z_inter_w = max(z_w_neg, z_w_pos);
    
    // LFVs from neighbour cells
    FlowVector Ustar_n_pos = coeffs_n.local_face_val(basis_n).get_star(z_n_pos, z_inter_n, solver_params.tol_h);
    FlowVector Ustar_e_pos = coeffs_e.local_face_val(basis_e).get_star(z_e_pos, z_inter_e, solver_params.tol_h);
    FlowVector Ustar_s_neg = coeffs_s.local_face_val(basis_s).get_star(z_s_neg, z_inter_s, solver_params.tol_h);
    FlowVector Ustar_w_neg = coeffs_w.local_face_val(basis_w).get_star(z_w_neg, z_inter_w, solver_params.tol_h);

    // LFVs of local cell
    FlowVector Ustar_n_neg = coeffs.local_face_val(basis_n_loc).get_star(z_n_neg, z_inter_n, solver_params.tol_h);
    FlowVector Ustar_e_neg = coeffs.local_face_val(basis_e_loc).get_star(z_e_neg, z_inter_e, solver_params.tol_h);
    FlowVector Ustar_s_pos = coeffs.local_face_val(basis_s_loc).get_star(z_s_pos, z_inter_s, solver_params.tol_h);
    FlowVector Ustar_w_pos = coeffs.local_face_val(basis_w_loc).get_star(z_w_pos, z_inter_w, solver_params.tol_h);

    FlowVector F_n = flux_HLL_y(Ustar_n_neg, Ustar_n_pos, solver_params, sim_params);
    FlowVector F_e = flux_HLL_x(Ustar_e_neg, Ustar_e_pos, solver_params, sim_params);
    FlowVector F_s = flux_HLL_y(Ustar_s_neg, Ustar_s_pos, solver_params, sim_params);
    FlowVector F_w = flux_HLL_x(Ustar_w_neg, Ustar_w_pos, solver_params, sim_params);
    
    FlowVector U0x_star = (Ustar_e_neg + Ustar_w_pos) /   C(2.0);
    FlowVector U1x_star = (Ustar_e_neg - Ustar_w_pos) / ( C(2.0) * sqrt( C(3.0) ) );

    FlowCoeffs Lx = {};

    Lx.set_0(-C(1.0) * (F_e - F_w) / dx_loc);

    Lx.set_1x
    (
        -sqrt( C(3.0) ) / dx_loc *
        (
            F_e + F_w
            - (U0x_star - U1x_star).phys_flux_x(solver_params.tol_h, sim_params.g)
            - (U0x_star + U1x_star).phys_flux_x(solver_params.tol_h, sim_params.g)
        )
    );

    Lx += get_bed_src_x
    (
        coeffs.local_face_val(basis_e_loc).h + z_e_neg,
        coeffs.local_face_val(basis_w_loc).h + z_w_pos,
        z_inter_e,
        z_inter_w,
        U0x_star.h,
        U1x_star.h,
        sim_params.g,
        dx_loc,
        coeffs,
        idx
    );

    FlowVector U0y_star = (Ustar_n_neg + Ustar_s_pos) /   C(2.0);
    FlowVector U1y_star = (Ustar_n_neg - Ustar_s_pos) / ( C(2.0) * sqrt(C(3.0) ) );

    FlowCoeffs Ly = {};

    Ly.set_0(-C(1.0) * (F_n - F_s) / dy_loc);

    Ly.set_1y
    (
        -sqrt( C(3.0) ) / dy_loc *
        (
            F_s + F_n
            - (U0y_star - U1y_star).phys_flux_y(solver_params.tol_h, sim_params.g)
            - (U0y_star + U1y_star).phys_flux_y(solver_params.tol_h, sim_params.g)
        )
    );
    
    Ly += get_bed_src_y
    (
        coeffs.local_face_val(basis_n_loc).h + z_n_neg,
        coeffs.local_face_val(basis_s_loc).h + z_s_pos,
        z_inter_n,
        z_inter_s,
        U0y_star.h,
        U1y_star.h,
        sim_params.g,
        dy_loc,
        coeffs
    );

    coeffs += dt * (Lx + Ly);

    real& h0   = d_assem_sol_store.h0[idx]  ;
    real& h1x  = d_assem_sol_store.h1x[idx] ;
    real& h1y  = d_assem_sol_store.h1y[idx] ;
    real& qx0  = d_assem_sol_store.qx0[idx] ;
    real& qx1x = d_assem_sol_store.qx1x[idx];
    real& qx1y = d_assem_sol_store.qx1y[idx];
    real& qy0  = d_assem_sol_store.qy0[idx] ;
    real& qy1x = d_assem_sol_store.qy1x[idx];
    real& qy1y = d_assem_sol_store.qy1y[idx];

    h0   = (rkdg2) ? C(0.5) * (coeffs.h._0   + h0  ) : coeffs.h._0  ;
    h1x  = (rkdg2) ? C(0.5) * (coeffs.h._1x  + h1x ) : coeffs.h._1x ;
    h1y  = (rkdg2) ? C(0.5) * (coeffs.h._1y  + h1y ) : coeffs.h._1y ;
    qx0  = (rkdg2) ? C(0.5) * (coeffs.qx._0  + qx0 ) : coeffs.qx._0 ;
    qx1x = (rkdg2) ? C(0.5) * (coeffs.qx._1x + qx1x) : coeffs.qx._1x;
    qx1y = (rkdg2) ? C(0.5) * (coeffs.qx._1y + qx1y) : coeffs.qx._1y;
    qy0  = (rkdg2) ? C(0.5) * (coeffs.qy._0  + qy0 ) : coeffs.qy._0 ;
    qy1x = (rkdg2) ? C(0.5) * (coeffs.qy._1x + qy1x) : coeffs.qy._1x;
    qy1y = (rkdg2) ? C(0.5) * (coeffs.qy._1y + qy1y) : coeffs.qy._1y;
    
    const bool below_depth = (h0 < solver_params.tol_h);

    if (below_depth)
    {
        qx0  = C(0.0);
        qx1x = C(0.0);
        qx1y = C(0.0);
        
        qy0  = C(0.0);
        qy1x = C(0.0);
        qy1y = C(0.0);
    }

    if (!below_depth)
    {
        real ux = qx0 / h0;
        real uy = qy0 / h0;
        
        real dt_x = solver_params.CFL * dx_loc / ( abs(ux) + sqrt(sim_params.g * h0) );
        real dt_y = solver_params.CFL * dy_loc / ( abs(uy) + sqrt(sim_params.g * h0) );

        d_dt_CFL[idx] = min(dt_x, dt_y);
    }
}