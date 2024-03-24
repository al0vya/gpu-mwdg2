#include "dg2_update_x.cuh"

__global__
void dg2_update_x
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

        d_dt_CFL[idx] = solver_params.initial_tstep;
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
    
    int level_e = d_neighbours.east.levels[idx];
    int level_w = d_neighbours.west.levels[idx];

    HierarchyIndex h_idx_e = d_neighbours.east.act_idcs[idx];
    HierarchyIndex h_idx_w = d_neighbours.west.act_idcs[idx];

    // between local && east,  x, y unit is (1.0, 0.5)
    // between local && west,  x, y unit is (0.0, 0.5)
    
    LegendreBasis basis_e = get_leg_basis(h_idx, h_idx_e, level_e, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, EAST);
    LegendreBasis basis_w = get_leg_basis(h_idx, h_idx_w, level_w, solver_params.L, x, y, dx_loc, dy_loc, dx_finest, dy_finest, WEST);
    
    LegendreBasis basis_e_loc = { C(1.0), sqrt( C(3.0) ), C(0.0) } ;
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
    PlanarCoefficients z_planar_e = { d_neighbours.east.z0[idx],  d_neighbours.east.z1x[idx],  d_neighbours.east.z1y[idx] };
    PlanarCoefficients z_planar_w = { d_neighbours.west.z0[idx],  d_neighbours.west.z1x[idx],  d_neighbours.west.z1y[idx] };

    // LFVs from neighbour cells
    real z_e_pos = eval_loc_face_val_dg2(z_planar_e, basis_e);
    real z_w_neg = eval_loc_face_val_dg2(z_planar_w, basis_w);

    // LFVs of local cell
    real z_e_neg = eval_loc_face_val_dg2(z_planar, basis_e_loc);
    real z_w_pos = eval_loc_face_val_dg2(z_planar, basis_w_loc);

    real z_inter_e = max(z_e_neg, z_e_pos);
    real z_inter_w = max(z_w_neg, z_w_pos);
    
    // LFVs from neighbour cells
    FlowVector Ustar_e_pos = coeffs_e.local_face_val(basis_e).get_star(z_e_pos, z_inter_e, solver_params.tol_h);
    FlowVector Ustar_w_neg = coeffs_w.local_face_val(basis_w).get_star(z_w_neg, z_inter_w, solver_params.tol_h);

    // LFVs of local cell
    FlowVector Ustar_e_neg = coeffs.local_face_val(basis_e_loc).get_star(z_e_neg, z_inter_e, solver_params.tol_h);
    FlowVector Ustar_w_pos = coeffs.local_face_val(basis_w_loc).get_star(z_w_pos, z_inter_w, solver_params.tol_h);

    FlowVector F_e = flux_HLL_x(Ustar_e_neg, Ustar_e_pos, solver_params, sim_params);
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

    coeffs += dt * Lx;

    d_assem_sol_store.h0[idx]   = coeffs.h._0  ;
    d_assem_sol_store.h1x[idx]  = coeffs.h._1x ;
    d_assem_sol_store.h1y[idx]  = coeffs.h._1y ;
    d_assem_sol_store.qx0[idx]  = coeffs.qx._0 ;
    d_assem_sol_store.qx1x[idx] = coeffs.qx._1x;
    d_assem_sol_store.qx1y[idx] = coeffs.qx._1y;
    d_assem_sol_store.qy0[idx]  = coeffs.qy._0 ;
    d_assem_sol_store.qy1x[idx] = coeffs.qy._1x;
    d_assem_sol_store.qy1y[idx] = coeffs.qy._1y;
}