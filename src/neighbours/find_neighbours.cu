#include "find_neighbours.cuh"

__global__
void find_neighbours
(
    AssembledSolution    d_assem_sol,
    Neighbours           d_neighbours,
    SimulationParams sim_params,
    int                  mesh_dim
)
{
    HierarchyIndex t_idx = threadIdx.x;
    HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

    Coordinate x = idx % mesh_dim;
    Coordinate y = idx / mesh_dim;

    HierarchyIndex north;
    HierarchyIndex east;
    HierarchyIndex south;
    HierarchyIndex west;

    HierarchyIndex active_idx_local;
    HierarchyIndex active_idx_north;
    HierarchyIndex active_idx_east;
    HierarchyIndex active_idx_south;
    HierarchyIndex active_idx_west;

    int lvl_local = d_assem_sol.levels[idx];
    int lvl_north;
    int lvl_east;
    int lvl_south;
    int lvl_west;

    HierarchyIndex lvl_idx_local;
    HierarchyIndex lvl_idx_north;
    HierarchyIndex lvl_idx_east;
    HierarchyIndex lvl_idx_south;
    HierarchyIndex lvl_idx_west;

    MortonCode code_north;
    MortonCode code_east;
    MortonCode code_south;
    MortonCode code_west;

    int lvl_diff_north;
    int lvl_diff_east;
    int lvl_diff_south;
    int lvl_diff_west;

    bool northern = ( y == (mesh_dim - 1) || y == (sim_params.ysz - 1) );
    bool eastern  = ( x == (mesh_dim - 1) || x == (sim_params.xsz - 1) );
    bool southern = ( y == 0 );
    bool western  = ( x == 0 );

    bool at_border = (northern || eastern || southern || western);

    if (!at_border)
    {
        north = idx + mesh_dim;
        east  = idx + 1;
        south = idx - mesh_dim;
        west  = idx - 1;

        active_idx_local = d_assem_sol.act_idcs[idx];
        active_idx_north = d_assem_sol.act_idcs[north];
        active_idx_east  = d_assem_sol.act_idcs[east];
        active_idx_south = d_assem_sol.act_idcs[south];
        active_idx_west  = d_assem_sol.act_idcs[west];

        lvl_north = d_assem_sol.levels[north];
        lvl_east  = d_assem_sol.levels[east];
        lvl_south = d_assem_sol.levels[south];
        lvl_west  = d_assem_sol.levels[west];

        lvl_idx_local = get_lvl_idx(lvl_local);
        lvl_idx_north = get_lvl_idx(lvl_north);
        lvl_idx_east  = get_lvl_idx(lvl_east);
        lvl_idx_south = get_lvl_idx(lvl_south);
        lvl_idx_west  = get_lvl_idx(lvl_west);

        code_north = active_idx_north - lvl_idx_north;
        code_east  = active_idx_east  - lvl_idx_east;
        code_south = active_idx_south - lvl_idx_south;
        code_west  = active_idx_west  - lvl_idx_west;

        if (
            lvl_north == 3 &&
            lvl_east == 5 &&
            lvl_south == 4 &&
            lvl_west == 5
            )
        {
            printf("active_idx_north: %d\n", active_idx_north);
            printf("active_idx_east : %d\n", active_idx_east);
            printf("active_idx_south: %d\n", active_idx_south);
            printf("active_idx_west : %d\n", active_idx_west);
        }

        // lvl_idx_east - lvl_idx_local is positive only if
        // eastern neighbour is finer, which is the only time
        // we want to shift; idx should stay the same otherwise
        lvl_diff_north = max(0, lvl_north - lvl_local);
        lvl_diff_east  = max(0, lvl_east  - lvl_local);
        lvl_diff_south = max(0, lvl_south - lvl_local);
        lvl_diff_west  = max(0, lvl_west  - lvl_local);

        // for the case where the eastern neighbour is coarser
        // lvl_idx_local would evidently be finer and hence bigger
        // we would not use this hence min to select lvl_idx_east
        // for other cases, we would always use lvl_idx_local
        // where e.g. eastern neighbour is finer or same resolution
        d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
        d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) );
        d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );
        d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

        d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
        d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
        d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);
        d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

        /*n_idx = min(lvl_idx_local, lvl_idx_north) + code_north >> (2 * lvl_diff_north);
        e_idx = min(lvl_idx_local, lvl_idx_east) + ( code_east >> (2 * lvl_diff_east) );
        s_idx = min(lvl_idx_local, lvl_idx_south) + code_south >> (2 * lvl_diff_south);
        w_idx = min(lvl_idx_local, lvl_idx_west) + code_west >> (2 * lvl_diff_west);

        d_neighbours.north.levels[idx] = get_level(n_idx);
        d_neighbours.east.levels[idx] = get_level(e_idx);
        d_neighbours.south.levels[idx] = get_level(s_idx);
        d_neighbours.west.levels[idx] = get_level(w_idx);*/
    }
    else
    {        
        int h_dir = (western)  ? WEST  : (eastern)  ? EAST  : 0;
        int v_dir = (southern) ? SOUTH : (northern) ? NORTH : 0;

        int position = v_dir | h_dir;

        // switch logic explained in Directions enum, Directions.h
        switch (position)
        {
        case NORTH:  // north
            d_neighbours.north.act_idcs[idx] = -1;
            d_neighbours.north.levels[idx] = lvl_local;

            east  = idx + 1;
            south = idx - mesh_dim;
            west  = idx - 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_east  = d_assem_sol.act_idcs[east];
            active_idx_south = d_assem_sol.act_idcs[south];
            active_idx_west  = d_assem_sol.act_idcs[west];

            lvl_east  = d_assem_sol.levels[east];
            lvl_south = d_assem_sol.levels[south];
            lvl_west  = d_assem_sol.levels[west];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_east  = get_lvl_idx(lvl_east);
            lvl_idx_south = get_lvl_idx(lvl_south);
            lvl_idx_west  = get_lvl_idx(lvl_west);

            code_east  = active_idx_east  - lvl_idx_east;
            code_south = active_idx_south - lvl_idx_south;
            code_west  = active_idx_west  - lvl_idx_west;

            lvl_diff_east  = max(0, lvl_east  - lvl_local);
            lvl_diff_south = max(0, lvl_south - lvl_local);
            lvl_diff_west  = max(0, lvl_west  - lvl_local);

            d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) ) ;
            d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );
            d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

            d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
            d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);
            d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

            break;
        case EAST:  // east
            d_neighbours.east.act_idcs[idx] = -1;
            d_neighbours.east.levels[idx] = lvl_local;

            north = idx + mesh_dim;
            south = idx - mesh_dim;
            west  = idx - 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_north = d_assem_sol.act_idcs[north];
            active_idx_south = d_assem_sol.act_idcs[south];
            active_idx_west  = d_assem_sol.act_idcs[west];
    
            lvl_north = d_assem_sol.levels[north];
            lvl_south = d_assem_sol.levels[south];
            lvl_west  = d_assem_sol.levels[west];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_north = get_lvl_idx(lvl_north);
            lvl_idx_south = get_lvl_idx(lvl_south);
            lvl_idx_west  = get_lvl_idx(lvl_west);

            code_north = active_idx_north - lvl_idx_north;
            code_south = active_idx_south - lvl_idx_south;
            code_west  = active_idx_west  - lvl_idx_west;

            lvl_diff_north = max(0, lvl_north - lvl_local);
            lvl_diff_south = max(0, lvl_south - lvl_local);
            lvl_diff_west  = max(0, lvl_west  - lvl_local);

            d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
            d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );
            d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

            d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
            d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);
            d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

            break;
        case SOUTH:  // south
            d_neighbours.south.act_idcs[idx] = -1;
            d_neighbours.south.levels[idx] = lvl_local;

            north = idx + mesh_dim;
            east  = idx + 1;
            west  = idx - 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_north = d_assem_sol.act_idcs[north];
            active_idx_east  = d_assem_sol.act_idcs[east];
            active_idx_west  = d_assem_sol.act_idcs[west];

            lvl_north = d_assem_sol.levels[north];
            lvl_east  = d_assem_sol.levels[east];
            lvl_west  = d_assem_sol.levels[west];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_north = get_lvl_idx(lvl_north);
            lvl_idx_east  = get_lvl_idx(lvl_east);
            lvl_idx_west  = get_lvl_idx(lvl_west);

            code_north = active_idx_north - lvl_idx_north;
            code_east  = active_idx_east  - lvl_idx_east;
            code_west  = active_idx_west  - lvl_idx_west;

            lvl_diff_north = max(0, lvl_north - lvl_local);
            lvl_diff_east  = max(0, lvl_east  - lvl_local);
            lvl_diff_west  = max(0, lvl_west  - lvl_local);

            d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
            d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) );
            d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

            d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
            d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
            d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

            break;
        case WEST:  // west
            d_neighbours.west.act_idcs[idx] = -1;
            d_neighbours.west.levels[idx] = lvl_local;

            north = idx + mesh_dim;
            east  = idx + 1;
            south = idx - mesh_dim;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_north = d_assem_sol.act_idcs[north];
            active_idx_east  = d_assem_sol.act_idcs[east];
            active_idx_south = d_assem_sol.act_idcs[south];

            lvl_north = d_assem_sol.levels[north];
            lvl_east  = d_assem_sol.levels[east];
            lvl_south = d_assem_sol.levels[south];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_north = get_lvl_idx(lvl_north);
            lvl_idx_east  = get_lvl_idx(lvl_east);
            lvl_idx_south = get_lvl_idx(lvl_south);

            code_north = active_idx_north - lvl_idx_north;
            code_east  = active_idx_east  - lvl_idx_east;
            code_south = active_idx_south - lvl_idx_south;

            lvl_diff_north = max(0, lvl_north - lvl_local);
            lvl_diff_east  = max(0, lvl_east  - lvl_local);
            lvl_diff_south = max(0, lvl_south - lvl_local);

            d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
            d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) );
            d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );

            d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
            d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
            d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);

            break;
        case NORTH | EAST:  // northeast
            d_neighbours.north.act_idcs[idx] = -1;
            d_neighbours.east.act_idcs[idx] = -1;

            d_neighbours.north.levels[idx] = lvl_local;
            d_neighbours.east.levels[idx] = lvl_local;

            south = idx - mesh_dim;
            west  = idx - 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_south = d_assem_sol.act_idcs[south];
            active_idx_west  = d_assem_sol.act_idcs[west];

    
            lvl_south = d_assem_sol.levels[south];
            lvl_west  = d_assem_sol.levels[west];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_south = get_lvl_idx(lvl_south);
            lvl_idx_west  = get_lvl_idx(lvl_west);

            code_south = active_idx_south - lvl_idx_south;
            code_west  = active_idx_west  - lvl_idx_west;

            lvl_diff_south = max(0, lvl_south - lvl_local);
            lvl_diff_west  = max(0, lvl_west  - lvl_local);

            d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );
            d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

            d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);
            d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

            break;
        case NORTH | WEST:  // northwest
            d_neighbours.north.act_idcs[idx] = -1;
            d_neighbours.west.act_idcs[idx] = -1;

            d_neighbours.north.levels[idx] = lvl_local;
            d_neighbours.west.levels[idx] = lvl_local;

            east  = idx + 1;
            south = idx - mesh_dim;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_east  = d_assem_sol.act_idcs[east];
            active_idx_south = d_assem_sol.act_idcs[south];

    
            lvl_east  = d_assem_sol.levels[east];
            lvl_south = d_assem_sol.levels[south];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_east  = get_lvl_idx(lvl_east);
            lvl_idx_south = get_lvl_idx(lvl_south);

            code_east  = active_idx_east  - lvl_idx_east;
            code_south = active_idx_south - lvl_idx_south;

            lvl_diff_east  = max(0, lvl_east  - lvl_local);
            lvl_diff_south = max(0, lvl_south - lvl_local);

            d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) );
            d_neighbours.south.act_idcs[idx] = min(lvl_idx_local, lvl_idx_south) + ( code_south >> (2 * lvl_diff_south) );

            d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
            d_neighbours.south.levels[idx] = min(lvl_local, lvl_south);

            break;
        case SOUTH | EAST:  // southeast
            d_neighbours.south.act_idcs[idx] = -1;
            d_neighbours.east.act_idcs[idx] = -1;

            d_neighbours.south.levels[idx] = lvl_local;
            d_neighbours.east.levels[idx] = lvl_local;

            north = idx + mesh_dim;
            west  = idx - 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_north = d_assem_sol.act_idcs[north];
            active_idx_west  = d_assem_sol.act_idcs[west];

    
            lvl_north = d_assem_sol.levels[north];
            lvl_west  = d_assem_sol.levels[west];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_north = get_lvl_idx(lvl_north);
            lvl_idx_west  = get_lvl_idx(lvl_west);

            code_north = active_idx_north - lvl_idx_north;
            code_west  = active_idx_west  - lvl_idx_west;

            lvl_diff_north = max(0, lvl_north - lvl_local);
            lvl_diff_west  = max(0, lvl_west  - lvl_local);

            d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
            d_neighbours.west.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_west)  + ( code_west  >> (2 * lvl_diff_west) );

            d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
            d_neighbours.west.levels[idx]  = min(lvl_local, lvl_west);

            break;
        case SOUTH | WEST: // southwest
            d_neighbours.south.act_idcs[idx] = -1;
            d_neighbours.west.act_idcs[idx] = -1;

            d_neighbours.south.levels[idx] = lvl_local;
            d_neighbours.west.levels[idx] = lvl_local;

            north = idx + mesh_dim;
            east  = idx + 1;

            active_idx_local = d_assem_sol.act_idcs[idx];
            active_idx_north = d_assem_sol.act_idcs[north];
            active_idx_east  = d_assem_sol.act_idcs[east];

    
            lvl_north = d_assem_sol.levels[north];
            lvl_east  = d_assem_sol.levels[east];

            lvl_idx_local = get_lvl_idx(lvl_local);
            lvl_idx_north = get_lvl_idx(lvl_north);
            lvl_idx_east  = get_lvl_idx(lvl_east);

            code_north = active_idx_north - lvl_idx_north;
            code_east  = active_idx_east  - lvl_idx_east;

            lvl_diff_north = max(0, lvl_north - lvl_local);
            lvl_diff_east  = max(0, lvl_east  - lvl_local);

            d_neighbours.north.act_idcs[idx] = min(lvl_idx_local, lvl_idx_north) + ( code_north >> (2 * lvl_diff_north) );
            d_neighbours.east.act_idcs[idx]  = min(lvl_idx_local, lvl_idx_east)  + ( code_east  >> (2 * lvl_diff_east) );

            d_neighbours.north.levels[idx] = min(lvl_local, lvl_north);
            d_neighbours.east.levels[idx]  = min(lvl_local, lvl_east);
            break;
        default:
            printf("ERROR: none of the directions were encountered. Thread: %d.\n", idx);
            return;
        }
    }
}