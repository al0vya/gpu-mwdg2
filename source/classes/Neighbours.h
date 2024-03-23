#pragma once

#include "../classes/AssembledSolution.h"

typedef struct Neighbours
{
    AssembledSolution north;
    AssembledSolution east;
    AssembledSolution south;
    AssembledSolution west;

    Neighbours
    (
        const SolverParams& solver_params
    )
    :
        north(solver_params, "north"),
        east (solver_params, "east"),
        south(solver_params, "south"),
        west (solver_params, "west")
    {}

    Neighbours
    (
        const SolverParams& solver_params,
        const char*         dirroot,
		const char*         prefix
    )
    :
        north(solver_params, "north", dirroot, prefix),
        east (solver_params, "east" , dirroot, prefix),
        south(solver_params, "south", dirroot, prefix),
        west (solver_params, "west" , dirroot, prefix)
    {}

    void write_to_file
    (
        const char* dirroot,
        const char* prefix
    )
    {
        this->north.write_to_file(dirroot, prefix);
        this->east.write_to_file (dirroot, prefix);
        this->south.write_to_file(dirroot, prefix);
        this->west.write_to_file (dirroot, prefix);
    }
    
    real verify_real
    (
        const char* dirroot,
        const char* prefix
    )
    {
        const real error_north = this->north.verify_real(dirroot, prefix);
        const real error_east  = this->east.verify_real (dirroot, prefix);
        const real error_south = this->south.verify_real(dirroot, prefix);
        const real error_west  = this->west.verify_real (dirroot, prefix);

        return std::max({ error_north, error_east, error_south, error_west });
    }

    int verify_int
    (
        const char* dirroot,
        const char* prefix
    )
    {
        const int diffs_north = this->north.verify_int(dirroot, prefix);
        const int diffs_east  = this->east.verify_int (dirroot, prefix);
        const int diffs_south = this->south.verify_int(dirroot, prefix);
        const int diffs_west  = this->west.verify_int (dirroot, prefix);

        return diffs_north + diffs_east + diffs_south + diffs_west;
    }

} Neighbours;