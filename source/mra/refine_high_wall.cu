#include "refine_high_wall.cuh"

__host__
void refine_high_wall
(
	const SimulationParams& sim_params,
	const int               max_ref_lvl,
	      bool*             h_preflagged_details
)
{
	const int refinement_thickness = 8;

	const int num_refined_cells_x = refinement_thickness * sim_params.xsz;
	const int num_refined_cells_y = refinement_thickness * sim_params.ysz;
	const int num_refined_cells = num_refined_cells_x + num_refined_cells_y;

	MortonCode* refined_high_wall_codes = new MortonCode[num_refined_cells];

	Coordinate x_max = sim_params.xsz - 1;
	Coordinate y_max = sim_params.ysz - 1;

	// refining the northern wall
	for (int j = 0; j < refinement_thickness; j++)
	{
		for (int i = 0; i < sim_params.xsz; i++)
		{
			Coordinate x = i;
			Coordinate y = y_max - j;

			refined_high_wall_codes[j * sim_params.xsz + i] = generate_morton_code(x, y);
		}
	}

	// refining the eastern wall
	for (int j = 0; j < refinement_thickness; j++)
	{
		for (int i = 0; i < sim_params.ysz; i++)
		{
			Coordinate x = x_max - j;
			Coordinate y = i;

			refined_high_wall_codes[num_refined_cells_x + j * sim_params.ysz + i] = generate_morton_code(x, y);
		}
	}

	HierarchyIndex starting_idx = get_lvl_idx(max_ref_lvl - 1);

	for (int i = 0; i < num_refined_cells; i++)
	{
		MortonCode child_idx = refined_high_wall_codes[i] / 4; // to get Morton code one level below

		h_preflagged_details[starting_idx + child_idx] = true;
	}

	delete[] refined_high_wall_codes;
}