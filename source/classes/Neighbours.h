#pragma once

#include "../classes/AssembledSolution.h"

typedef struct Neighbours
{
    AssembledSolution north;
    AssembledSolution east;
    AssembledSolution south;
    AssembledSolution west;

    Neighbours(const SolverParams& solver_params)
    :
        north(solver_params),
        east (solver_params),
        south(solver_params),
        west (solver_params)
    {}

} Neighbours;