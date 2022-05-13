#include "get_child_details.cuh"

__device__
DetailChildren get_child_details
(
	bool* shared_child_details,
	int   t_idx
)
{
	return
	{
		shared_child_details[t_idx + 0], // detail_0
		shared_child_details[t_idx + 1], // detail_1
		shared_child_details[t_idx + 2], // detail_2
		shared_child_details[t_idx + 3]  // detail_3
	};
}