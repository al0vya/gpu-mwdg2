#pragma once

#include "AssembledSolution.h"

typedef struct Neighbours
{
    AssembledSolution north;
    AssembledSolution east;
    AssembledSolution south;
    AssembledSolution west;

    Neighbours(const int& num_finest_elems, const int& solver_type)
    :
        north(num_finest_elems, solver_type),
        east (num_finest_elems, solver_type),
        south(num_finest_elems, solver_type),
        west (num_finest_elems, solver_type)
    {}

} Neighbours;