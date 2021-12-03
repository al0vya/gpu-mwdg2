#include "fv1_update.cuh"

__global__
void fv1_update
(
    Neighbours           d_neighbours,
    AssembledSolution    d_assem_sol,
    SolverParameters     solver_params,
    SimulationParameters sim_params,
    real                 dx_finest,
    real                 dy_finest,
    real                 dt,
    real*                d_dt_CFL
)
{
    typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

    __shared__ union
    {
        typename block_scan::TempStorage temp_storage;
        HierarchyIndex indices[THREADS_PER_BLOCK];

    } shared;

    HierarchyIndex t_idx = threadIdx.x;
    HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;
    
    int is_wet = 0;

    int thread_prefix_sum = 0;

    int num_wet = 0;

    if (idx < d_assem_sol.length)
    {
        real h   = d_assem_sol.h0[idx];
        real h_n = d_neighbours.north.h0[idx];
        real h_e = d_neighbours.east.h0[idx];
        real h_s = d_neighbours.south.h0[idx];
        real h_w = d_neighbours.west.h0[idx];

        is_wet =
        (
            h   >= solver_params.tol_h ||
            h_n >= solver_params.tol_h ||
            h_e >= solver_params.tol_h ||
            h_s >= solver_params.tol_h ||
            h_w >= solver_params.tol_h
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
    
    int level = d_assem_sol.levels[idx];

    real dx_loc = dx_finest * ( 1 << (solver_params.L - level) );
    real dy_loc = dy_finest * ( 1 << (solver_params.L - level) );

    real& z  = d_assem_sol.z0[idx];
    real& h  = d_assem_sol.h0[idx];
    real& qx = d_assem_sol.qx0[idx];
    real& qy = d_assem_sol.qy0[idx];

    real eta_n_neg = h + z;
    real eta_e_neg = h + z;
    real eta_s_pos = h + z;
    real eta_w_pos = h + z;

    real h_star_n_neg;
    real h_star_e_neg;
    real h_star_s_pos;
    real h_star_w_pos;

    real z_n = d_neighbours.north.z0[idx];
    real z_e = d_neighbours.east.z0[idx];
    real z_s = d_neighbours.south.z0[idx];
    real z_w = d_neighbours.west.z0[idx];

    real z_n_intermediate = max(z, z_n);
    real z_e_intermediate = max(z, z_e);
    real z_s_intermediate = max(z, z_s);
    real z_w_intermediate = max(z, z_w);

    // northern flux
    FlowVector F_n;
    {
        FlowVector U_n_neg =
        {
            h,
            qx,
            qy
        };

        FlowVector U_star_n_neg = U_n_neg.get_star(z, z_n_intermediate, solver_params.tol_h);

        h_star_n_neg = U_star_n_neg.h;

        FlowVector U_n_pos =
        {
            d_neighbours.north.h0[idx],
            d_neighbours.north.qx0[idx],
            d_neighbours.north.qy0[idx]
        };

        FlowVector U_star_n_pos = U_n_pos.get_star(z_n, z_n_intermediate, solver_params.tol_h);
        
        F_n = flux_HLL_y
        (
            U_star_n_neg, 
            U_star_n_pos, 
            solver_params, 
            sim_params
        );
    }

    // eastern flux
    FlowVector F_e;
    {
        FlowVector U_e_neg =
        {
            h,
            qx,
            qy
        };

        FlowVector U_star_e_neg = U_e_neg.get_star(z, z_e_intermediate, solver_params.tol_h);

        h_star_e_neg = U_star_e_neg.h;

        FlowVector U_e_pos =
        {
            d_neighbours.east.h0[idx],
            d_neighbours.east.qx0[idx],
            d_neighbours.east.qy0[idx]
        };

        FlowVector U_star_e_pos = U_e_pos.get_star(z_e, z_e_intermediate, solver_params.tol_h);

        F_e = flux_HLL_x
        (
            U_star_e_neg, 
            U_star_e_pos, 
            solver_params, 
            sim_params
        );
    }

    // southern flux
    FlowVector F_s;
    {
        FlowVector U_s_neg =
        {
            d_neighbours.south.h0[idx],
            d_neighbours.south.qx0[idx],
            d_neighbours.south.qy0[idx]
        };

        FlowVector U_star_s_neg = U_s_neg.get_star(z_s, z_s_intermediate, solver_params.tol_h);

        FlowVector U_s_pos =
        {
            h,
            qx,
            qy
        };

        FlowVector U_star_s_pos = U_s_pos.get_star(z, z_s_intermediate, solver_params.tol_h);

        h_star_s_pos = U_star_s_pos.h;

        F_s = flux_HLL_y
        (
            U_star_s_neg, 
            U_star_s_pos, 
            solver_params, 
            sim_params
        );
    }
    
    // western flux
    FlowVector F_w;
    {
        FlowVector U_w_neg =
        {
            d_neighbours.west.h0[idx],
            d_neighbours.west.qx0[idx],
            d_neighbours.west.qy0[idx]
        };

        FlowVector U_star_w_neg = U_w_neg.get_star(z_w, z_w_intermediate, solver_params.tol_h);

        FlowVector U_w_pos =
        {
            h,
            qx,
            qy
        };

        FlowVector U_star_w_pos = U_w_pos.get_star(z, z_w_intermediate, solver_params.tol_h);

        h_star_w_pos = U_star_w_pos.h;

        F_w = flux_HLL_x
        (
            U_star_w_neg, 
            U_star_w_pos, 
            solver_params, 
            sim_params
        );
    }

    real bed_src_x = get_bed_src_x
    (
        z_w_intermediate, 
        z_e_intermediate, 
        h_star_w_pos, 
        h_star_e_neg, 
        eta_w_pos, 
        eta_e_neg, 
        sim_params.g, 
        dx_loc,
        idx
    );

    real bed_src_y = get_bed_src_y
    (
        z_s_intermediate, 
        z_n_intermediate, 
        h_star_s_pos, 
        h_star_n_neg, 
        eta_s_pos, 
        eta_n_neg,
        sim_params.g, 
        dy_loc
    );
    
    // x dir update
    h  = h  - dt * (F_e.h - F_w.h) / dx_loc;
    qx = qx - dt * ( (F_e.qx - F_w.qx) / dx_loc + bed_src_x );
    qy = qy - dt * (F_e.qy - F_w.qy) / dx_loc;

    // y dir update
    h  = h  - dt * (F_n.h - F_s.h) / dy_loc;
    qx = qx - dt * (F_n.qx - F_s.qx) / dy_loc;  
    qy = qy - dt * ( (F_n.qy - F_s.qy) / dy_loc + bed_src_y );

    bool below_depth = (h < solver_params.tol_h);
    bool below_disch = (abs(qx) < solver_params.tol_q && abs(qy) < solver_params.tol_q);

    bool below_thres = (below_depth || below_disch);

    if (below_thres)
    {
        qx = C(0.0);
        qy = C(0.0);
    }
    
    if (!below_depth)
    {
        real ux = qx / h;
        real uy = qy / h; 
        
        real dt_x = solver_params.CFL * dx_loc / ( abs(ux) + sqrt(sim_params.g * h) );
        real dt_y = solver_params.CFL * dy_loc / ( abs(uy) + sqrt(sim_params.g * h) );

        d_dt_CFL[idx] = min(dt_x, dt_y);
    }
}