#pragma once

enum Directions
{
    NORTH = 1, // 0001
    EAST  = 2, // 0010
    SOUTH = 4, // 0100
    WEST  = 8  // 1000

    // NORTH | EAST = 3
    // NORTH | WEST = 9
    // SOUTH | EAST = 6
    // SOUTH | WEST = 12
};