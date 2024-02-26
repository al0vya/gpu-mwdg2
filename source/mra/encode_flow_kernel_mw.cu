#include "encode_flow_kernel_mw.cuh"

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
void encode_flow_kernel_mw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real* d_norm_details,
	bool* d_sig_details,
	bool* d_preflagged_details,
	Maxes             maxes,
	SolverParams      solver_params,
	int               level,
	int               num_threads,
	bool              for_nghbrs
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;

	real norm_detail = C(0.0);
	real epsilon_local = solver_params.epsilon / (1 << (solver_params.L - level));

	HierarchyIndex prev_lvl_idx = get_lvl_idx(level - 1);
	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

	__shared__ union
	{
		typename block_scan::TempStorage temp_storage;
		real     coeffs[THREADS_PER_BLOCK];

	} shared;

	__shared__ HierarchyIndex parents[THREADS_PER_BLOCK];

	HierarchyIndex parent_idx = curr_lvl_idx + idx;
	HierarchyIndex child_idx = next_lvl_idx + 4 * idx;

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

	if (!for_nghbrs) d_sig_details[parent_idx] = INSIGNIFICANT;

	if (is_sig) parents[thread_prefix_sum] = parent_idx;

	__syncthreads();

	if (t_idx >= num_sig_details) return;

	parent_idx = parents[t_idx];

	child_idx = next_lvl_idx + 4 * (parent_idx - curr_lvl_idx);

	// Encoding eta
	real* s0 = &d_scale_coeffs.eta0[child_idx + 0];
	real* s1x = &d_scale_coeffs.eta1x[child_idx + 0];
	real* s1y = &d_scale_coeffs.eta1y[child_idx + 0];

	ScaleChildrenMW children =
	{
		{  s0[0],  s0[1],  s0[2],  s0[3] },
		{ s1x[0], s1x[1], s1x[2], s1x[3] },
		{ s1y[0], s1y[1], s1y[2], s1y[3] }
	};

	d_scale_coeffs.eta0[parent_idx] = encode_scale_0(children);
	d_scale_coeffs.eta1x[parent_idx] = encode_scale_1x(children);
	d_scale_coeffs.eta1y[parent_idx] = encode_scale_1y(children);

	SubDetailMW subdetail = encode_detail(children);

	store_details
	(
		d_details.eta0,
		d_details.eta1x,
		d_details.eta1y,
		subdetail,
		parent_idx
	);

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.eta);

	// encoding qx
	s0 = &d_scale_coeffs.qx0[child_idx + 0];
	s1x = &d_scale_coeffs.qx1x[child_idx + 0];
	s1y = &d_scale_coeffs.qx1y[child_idx + 0];

	children =
	{
		{  s0[0],  s0[1],  s0[2],  s0[3] },
		{ s1x[0], s1x[1], s1x[2], s1x[3] },
		{ s1y[0], s1y[1], s1y[2], s1y[3] }
	};

	d_scale_coeffs.qx0[parent_idx] = encode_scale_0(children);
	d_scale_coeffs.qx1x[parent_idx] = encode_scale_1x(children);
	d_scale_coeffs.qx1y[parent_idx] = encode_scale_1y(children);

	subdetail = encode_detail(children);

	store_details
	(
		d_details.qx0,
		d_details.qx1x,
		d_details.qx1y,
		subdetail,
		parent_idx
	);

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.qx);

	// encoding qy
	s0 = &d_scale_coeffs.qy0[child_idx + 0];
	s1x = &d_scale_coeffs.qy1x[child_idx + 0];
	s1y = &d_scale_coeffs.qy1y[child_idx + 0];

	children =
	{
		{  s0[0],  s0[1],  s0[2],  s0[3] },
		{ s1x[0], s1x[1], s1x[2], s1x[3] },
		{ s1y[0], s1y[1], s1y[2], s1y[3] }
	};

	d_scale_coeffs.qy0[parent_idx] = encode_scale_0(children);
	d_scale_coeffs.qy1x[parent_idx] = encode_scale_1x(children);
	d_scale_coeffs.qy1y[parent_idx] = encode_scale_1y(children);

	subdetail = encode_detail(children);

	store_details
	(
		d_details.qy0,
		d_details.qy1x,
		d_details.qy1y,
		subdetail,
		parent_idx
	);

	norm_detail = max(norm_detail, subdetail.get_max() / maxes.qy);

	if (!for_nghbrs)
	{
		d_norm_details[parent_idx] = norm_detail;

		d_sig_details[parent_idx] = (norm_detail >= epsilon_local || d_preflagged_details[parent_idx] == SIGNIFICANT)
			? SIGNIFICANT
			: INSIGNIFICANT;
	}
}