#include "encode_flow_kernel_single_block_hw.cuh"

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
void encode_flow_kernel_single_block_hw
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

	real tol_q = solver_params.tol_q;

	HierarchyIndex parent_idx = curr_lvl_idx + t_idx;
	HierarchyIndex child_idx  = next_lvl_idx + 4 * t_idx;

	bool is_sig = d_sig_details[parent_idx];

	if (is_sig)
	{
		real* eta = &d_scale_coeffs.eta0[child_idx];
		real* qx = &d_scale_coeffs.qx0[child_idx];
		real* qy = &d_scale_coeffs.qy0[child_idx];

		ChildScaleCoeffsHW child_coeffs =
		{
			{ eta[0], eta[1], eta[2], eta[3] },

			{
				(abs(qx[0]) > tol_q) ? qx[0] : C(0.0),
				(abs(qx[1]) > tol_q) ? qx[1] : C(0.0),
				(abs(qx[2]) > tol_q) ? qx[2] : C(0.0),
				(abs(qx[3]) > tol_q) ? qx[3] : C(0.0)
			},

			{
				(abs(qy[0]) > tol_q) ? qy[0] : C(0.0),
				(abs(qy[1]) > tol_q) ? qy[1] : C(0.0),
				(abs(qy[2]) > tol_q) ? qy[2] : C(0.0),
				(abs(qy[3]) > tol_q) ? qy[3] : C(0.0)
			},

			{ C(0.0), C(0.0), C(0.0), C(0.0) }
		};

		ParentScaleCoeffsHW parent_coeffs = encode_scale_coeffs(child_coeffs);
		DetailHW            detail = encode_details(child_coeffs);

		parent_coeffs.qx *= (abs(parent_coeffs.qx) > tol_q);
		parent_coeffs.qy *= (abs(parent_coeffs.qy) > tol_q);

		norm_detail = detail.get_norm_detail(maxes);

		store_scale_coeffs
		(
			parent_coeffs,
			d_scale_coeffs,
			parent_idx
		);

		store_details
		(
			detail,
			d_details,
			parent_idx
		);

		d_norm_details[parent_idx] = norm_detail;

		d_sig_details[parent_idx] = (norm_detail >= epsilon_local) ? SIGNIFICANT : INSIGNIFICANT;

		if (d_preflagged_details[parent_idx] == SIGNIFICANT) d_sig_details[parent_idx] = SIGNIFICANT;
	}
}