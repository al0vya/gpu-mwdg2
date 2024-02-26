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

	typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

	__shared__ union
	{
		typename block_scan::TempStorage temp_storage;
		real     coeffs[THREADS_PER_BLOCK];

	} shared;
	
	__shared__ HierarchyIndex parents[THREADS_PER_BLOCK]; 

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

	if (!for_nghbrs) d_sig_details[parent_idx] = INSIGNIFICANT;

	if (is_sig) parents[thread_prefix_sum] = parent_idx;

	__syncthreads();

	if (t_idx >= num_sig_details) return;

	parent_idx = parents[t_idx];

	HierarchyIndex shared_idx = 0;

	real eta[4];
	real  qx[4];
	real  qy[4];

#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		shared_idx = t_idx + i * num_sig_details;
		child_idx = next_lvl_idx + 4 * (parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

		// loading eta
		shared.coeffs[t_idx] = d_scale_coeffs.eta0[child_idx];
		__syncthreads();
#pragma unroll
		for (int j = 0; j < 4; j++) if ((4 * t_idx + j) / num_sig_details == i) eta[j] = shared.coeffs[4 * t_idx + j - (i * num_sig_details)];
		__syncthreads();

		// loading qx
		shared.coeffs[t_idx] = d_scale_coeffs.qx0[child_idx];
		__syncthreads();
#pragma unroll
		for (int j = 0; j < 4; j++) if ((4 * t_idx + j) / num_sig_details == i) qx[j] = shared.coeffs[4 * t_idx + j - (i * num_sig_details)];
		__syncthreads();

		// loading qy
		shared.coeffs[t_idx] = d_scale_coeffs.qy0[child_idx];
		__syncthreads();
#pragma unroll
		for (int j = 0; j < 4; j++) if ((4 * t_idx + j) / num_sig_details == i) qy[j] = shared.coeffs[4 * t_idx + j - (i * num_sig_details)];
		__syncthreads();
	}

	ChildScaleCoeffsHW child_coeffs =
	{
		{ eta[0], eta[1], eta[2], eta[3] },
		{ qx[0], qx[1], qx[2], qx[3] },
		{ qy[0], qy[1], qy[2], qy[3] },
		{ C(0.0), C(0.0), C(0.0), C(0.0) }
	};

	ParentScaleCoeffsHW parent_coeffs = encode_scale_coeffs(child_coeffs);
	DetailHW            detail = encode_details(child_coeffs);

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