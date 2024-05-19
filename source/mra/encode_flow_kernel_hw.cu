#include "encode_flow_kernel_hw.cuh"

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
void encode_flow_kernel_hw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	real              epsilon_local,
	HierarchyIndex    curr_lvl_idx,
	HierarchyIndex    next_lvl_idx,
	int               level,
	int               num_threads
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;
	
	real norm_detail = C(0.0);
	
	typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

	__shared__ union
	{
		typename block_scan::TempStorage temp_storage;
		HierarchyIndex parents[THREADS_PER_BLOCK];

	} shared;
	
	HierarchyIndex parent_idx = curr_lvl_idx + idx;
	HierarchyIndex child_idx  = next_lvl_idx + 4 * idx;

	int is_sig = d_sig_details[parent_idx];

	int thread_prefix_sum = 0;

	int num_sig_details = 0;

	block_scan(shared.temp_storage).ExclusiveSum
	(
		is_sig,
		thread_prefix_sum,
		num_sig_details
	);

	__syncthreads();

	d_sig_details[parent_idx] = INSIGNIFICANT;

	if (is_sig) shared.parents[thread_prefix_sum] = parent_idx;

	__syncthreads();

	if (t_idx >= num_sig_details) return;

	parent_idx = shared.parents[t_idx];
	
	child_idx = next_lvl_idx + 4 * (parent_idx - curr_lvl_idx);

	ScaleChildrenHW children;
	SubDetailHW     subdetail;

	// Encoding eta
	load_children_vector
	(
		children,
		d_scale_coeffs.eta0,
		child_idx
	);
	
	d_scale_coeffs.eta0[parent_idx] = encode_scale(children);
	
	subdetail =
	{
		encode_detail_alpha(children),
		encode_detail_beta(children),
		encode_detail_gamma(children)
	};

	d_details.eta0.alpha[parent_idx] = subdetail.alpha;
	d_details.eta0.beta [parent_idx] = subdetail.beta ;
	d_details.eta0.gamma[parent_idx] = subdetail.gamma;

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.eta);

	// encoding qx
	load_children_vector
	(
		children,
		d_scale_coeffs.qx0,
		child_idx
	);
	
	d_scale_coeffs.qx0[parent_idx] = encode_scale(children);
	
	subdetail =
	{
		encode_detail_alpha(children),
		encode_detail_beta(children),
		encode_detail_gamma(children)
	};

	d_details.qx0.alpha[parent_idx] = subdetail.alpha;
	d_details.qx0.beta [parent_idx] = subdetail.beta ;
	d_details.qx0.gamma[parent_idx] = subdetail.gamma;

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.qx);

	// encoding qy
	load_children_vector
	(
		children,
		d_scale_coeffs.qy0,
		child_idx
	);
	
	d_scale_coeffs.qy0[parent_idx] = encode_scale(children);
	
	subdetail =
	{
		encode_detail_alpha(children),
		encode_detail_beta(children),
		encode_detail_gamma(children)
	};

	d_details.qy0.alpha[parent_idx] = subdetail.alpha;
	d_details.qy0.beta [parent_idx] = subdetail.beta ;
	d_details.qy0.gamma[parent_idx] = subdetail.gamma;

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.qy);

	d_norm_details[parent_idx] = norm_detail;

	d_sig_details[parent_idx] = (norm_detail >= epsilon_local || d_preflagged_details[parent_idx] == SIGNIFICANT)
		? SIGNIFICANT
		: INSIGNIFICANT;
}