#include "preflag_details.cuh"

__host__
bool* preflag_details
(
	const Boundaries&   boundaries,
	const PointSources& point_sources,
	const GaugePoints&  gauge_points,
	const int&          num_details,
	const int&          max_ref_lvl
)
{
	size_t bytes = num_details * sizeof(bool);
	
	bool* h_preflagged_details = new bool[num_details]();

	bool* d_preflagged_details = (bool*)malloc_device(bytes);

	HierarchyIndex starting_idx = get_lvl_idx(max_ref_lvl - 1);

	for (int i = 0; i < gauge_points.num_points; i++)
	{
		MortonCode child_idx = gauge_points.codes[i] / 4; // to get Morton code one level below

		h_preflagged_details[starting_idx + child_idx] = true;
	}

	if (nullptr != boundaries.north.codes)
	{
		for (int i = 0; i < boundaries.north.num_cells(); i++)
		{
			MortonCode child_idx = boundaries.north.codes[i] / 4;

			h_preflagged_details[starting_idx + child_idx] = true;
		}
	}

	if (nullptr != boundaries.east.codes)
	{
		for (int i = 0; i < boundaries.east.num_cells(); i++)
		{
			MortonCode child_idx = boundaries.east.codes[i] / 4;

			h_preflagged_details[starting_idx + child_idx] = true;
		}
	}

	if (nullptr != boundaries.south.codes)
	{
		for (int i = 0; i < boundaries.south.num_cells(); i++)
		{
			MortonCode child_idx = boundaries.south.codes[i] / 4;

			h_preflagged_details[starting_idx + child_idx] = true;
		}
	}

	if (nullptr != boundaries.west.codes)
	{
		for (int i = 0; i < boundaries.west.num_cells(); i++)
		{
			MortonCode child_idx = boundaries.west.codes[i] / 4;

			h_preflagged_details[starting_idx + child_idx] = true;
		}
	}

	for (int i = 0; i < point_sources.num_srcs; i++)
	{
		MortonCode child_idx = point_sources.h_codes[i] / 4;

		h_preflagged_details[starting_idx + child_idx] = true;
	}

	copy(d_preflagged_details, h_preflagged_details, bytes);

	delete[] h_preflagged_details;

	return d_preflagged_details;
}