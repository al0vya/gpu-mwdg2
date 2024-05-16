#include "preflag_details.cuh"

__host__
bool* preflag_details
(
	const Boundaries&        boundaries,
	const PointSources&      point_sources,
	const StagePoints&       stage_points,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const int&               max_ref_lvl,
	const int&               test_case
)
{
	const int    num_details = get_lvl_idx(max_ref_lvl);
	const size_t bytes       = num_details * sizeof(bool);
	
	bool* h_preflagged_details = new bool[num_details]();
	bool* d_preflagged_details = (bool*)malloc_device(bytes);

	HierarchyIndex starting_idx = get_lvl_idx(max_ref_lvl - 1);

	if ( !( solver_params.epsilon > C(0.0) ) )
	{
		for (Coordinate j = 0; j < sim_params.ysz; j++)
		{
			for (Coordinate i = 0; i < sim_params.xsz; i++)
			{
				MortonCode child_code = generate_morton_code(i, j) / 4;

				h_preflagged_details[starting_idx + child_code] = true;
			}
		}
	}

	for (int i = 0; i < stage_points.num_points; i++)
	{
		MortonCode child_code = stage_points.codes[i] / 4; // to get Morton code one level below

		h_preflagged_details[starting_idx + child_code] = true;
	}

	if (nullptr != boundaries.north.codes)
	{
		for (int i = 0; i < boundaries.north.num_cells(); i++)
		{
			MortonCode child_code = boundaries.north.codes[i] / 4;

			h_preflagged_details[starting_idx + child_code] = true;
		}
	}

	if (nullptr != boundaries.east.codes)
	{
		for (int i = 0; i < boundaries.east.num_cells(); i++)
		{
			MortonCode child_code = boundaries.east.codes[i] / 4;

			h_preflagged_details[starting_idx + child_code] = true;
		}
	}

	if (nullptr != boundaries.south.codes)
	{
		for (int i = 0; i < boundaries.south.num_cells(); i++)
		{
			MortonCode child_code = boundaries.south.codes[i] / 4;

			h_preflagged_details[starting_idx + child_code] = true;
		}
	}

	if (nullptr != boundaries.west.codes)
	{
		for (int i = 0; i < boundaries.west.num_cells(); i++)
		{
			MortonCode child_code = boundaries.west.codes[i] / 4;

			h_preflagged_details[starting_idx + child_code] = true;
		}
	}

	for (int i = 0; i < point_sources.num_srcs; i++)
	{
		MortonCode child_code = point_sources.h_codes[i] / 4;

		h_preflagged_details[starting_idx + child_code] = true;
	}

	if (test_case == 0 && solver_params.refine_wall)
	{
		refine_high_wall
		(
			sim_params,
			solver_params,
			max_ref_lvl,
			h_preflagged_details
		);
	}

	copy_cuda(d_preflagged_details, h_preflagged_details, bytes);

	delete[] h_preflagged_details;

	return d_preflagged_details;
}