#include "decoding_kernel_hw.cuh"

__global__
void decoding_kernel_hw
(
	bool*             d_sig_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	HierarchyIndex    curr_lvl_idx,
	HierarchyIndex    next_lvl_idx,
	int               num_threads
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;

	typedef cub::BlockScan<int, THREADS_PER_BLOCK> block_scan;

	__shared__ union
	{
		typename block_scan::TempStorage temp_storage;
		HierarchyIndex parents[THREADS_PER_BLOCK];

	} shared;

	HierarchyIndex parent_idx = curr_lvl_idx + idx;

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

	if (is_sig) shared.parents[thread_prefix_sum] = parent_idx;

	__syncthreads();

	if (t_idx >= num_sig_details) return;

	parent_idx = shared.parents[t_idx];

	HierarchyIndex child_idx = next_lvl_idx + 4 * (parent_idx - curr_lvl_idx);
	
	real            parent_coeff;
	ScaleChildrenHW children;
	SubDetailHW     subdetails;

	// Decoding eta
	parent_coeff = d_scale_coeffs.eta0[parent_idx];

	subdetails = load_subdetails_hw
	(
		d_details.eta0,
		parent_idx
	);

	children = decode_scale_children(parent_coeff, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.eta0,
		child_idx
	);

	// Decoding qx
	parent_coeff = d_scale_coeffs.qx0[parent_idx];

	subdetails = load_subdetails_hw
	(
		d_details.qx0,
		parent_idx
	);

	children = decode_scale_children(parent_coeff, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.qx0,
		child_idx
	);

	// Decoding qy
	parent_coeff = d_scale_coeffs.qy0[parent_idx];

	subdetails = load_subdetails_hw
	(
		d_details.qy0,
		parent_idx
	);

	children = decode_scale_children(parent_coeff, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.qy0,
		child_idx
	);
}