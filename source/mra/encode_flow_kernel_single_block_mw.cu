#include "encode_flow_kernel_single_block_mw.cuh"

/*
 *
 * This kernel launches every refinement level starting from n = L, using 4^n threads.
 * These threads load the scale coefficients into shared memory.
 * These scale coefficients are the child coefficients of the sub-elements at n - 1.
 * At n - 1, there are 4^(n-1) sub-elements i.e. a quarter compared to 4^n.
 * Hence, after loading the scale coefficients into shared memory, only 4^(n-1) threads are kept.
 * Thereafter, each thread loads four child coefficients from shared memory to encode the parent and detail.
 * A block size that is a multiple of 4 is used to ensure enough child coefficients are loaded.
 * For the block sizes below, there is a refinement level at which only one thread block is launched:
 * 
 * Block size: 64.   Level at which only a single block is launched (LVL_SINGLE_BLOCK): 3.
 * Block size: 256.  Level at which only a single block is launched (LVL_SINGLE_BLOCK): 4.
 * Block size: 1024. Level at which only a single block is launched (LVL_SINGLE_BLOCK): 5.
 * 
 * In this scenario, the kernel is not relaunched, as a single block has enough threads for all subsequent levels.
 * Instead, there is an internal for-loop across levels, which writes the scale coefficients to shared memory.
 * The threads in the next iteration of the loop access the shared memory, which is visible to all threads within a block.
 * 
 */

__global__
void encode_flow_kernel_single_block_mw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams      solver_params,
	int               level,
	int               num_threads,
	bool              for_nghbrs
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;
	
	real norm_detail   = C(0.0);
	real epsilon_local = solver_params.epsilon / ( 1 << (solver_params.L - level) );

	HierarchyIndex prev_lvl_idx = get_lvl_idx(level - 1);
	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	HierarchyIndex parent_idx = curr_lvl_idx + t_idx;
	HierarchyIndex child_idx  = next_lvl_idx + 4 * t_idx;

	bool is_sig = d_sig_details[parent_idx];

	if (is_sig)
	{
		real* eta0 = &d_scale_coeffs.eta0[child_idx];
		real* qx0 = &d_scale_coeffs.qx0[child_idx];
		real* qy0 = &d_scale_coeffs.qy0[child_idx];

		real* eta1x = &d_scale_coeffs.eta1x[child_idx];
		real* qx1x = &d_scale_coeffs.qx1x[child_idx];
		real* qy1x = &d_scale_coeffs.qy1x[child_idx];

		real* eta1y = &d_scale_coeffs.eta1y[child_idx];
		real* qx1y = &d_scale_coeffs.qx1y[child_idx];
		real* qy1y = &d_scale_coeffs.qy1y[child_idx];

		ChildScaleCoeffsMW child_coeffs =
		{
			{
				{  eta0[0],  eta0[1],  eta0[2],  eta0[3] },
				{ eta1x[0], eta1x[1], eta1x[2], eta1x[3] },
				{ eta1y[0], eta1y[1], eta1y[2], eta1y[3] }
			},
			{
				{  qx0[0],  qx0[1],  qx0[2],  qx0[3] },
				{ qx1x[0], qx1x[1], qx1x[2], qx1x[3] },
				{ qx1y[0], qx1y[1], qx1y[2], qx1y[3] }
			},
			{
				{  qy0[0],  qy0[1],  qy0[2],  qy0[3] },
				{ qy1x[0], qy1x[1], qy1x[2], qy1x[3] },
				{ qy1y[0], qy1y[1], qy1y[2], qy1y[3] }
			},
			{
				{ C(0.0), C(0.0), C(0.0), C(0.0) },
				{ C(0.0), C(0.0), C(0.0), C(0.0) },
				{ C(0.0), C(0.0), C(0.0), C(0.0) }
			}
		};

		ParentScaleCoeffsMW parent_coeffs = encode_scale_coeffs(child_coeffs);
		DetailMW            detail = (!for_nghbrs) ? encode_details(child_coeffs) : DetailMW{};

		norm_detail = detail.get_norm_detail(maxes);

		store_scale_coeffs
		(
			parent_coeffs,
			d_scale_coeffs,
			parent_idx
		);

		if (!for_nghbrs)
		{
			store_details
			(
				detail,
				d_details,
				parent_idx
			);
		}
	}

	if (!for_nghbrs)
	{
		d_norm_details[parent_idx] = norm_detail;

		d_sig_details[parent_idx] = (norm_detail >= epsilon_local) ? SIGNIFICANT : INSIGNIFICANT;

		if (d_preflagged_details[parent_idx] == SIGNIFICANT) d_sig_details[parent_idx] = SIGNIFICANT;
	}
}