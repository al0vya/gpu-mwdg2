#include "decoding_kernel_mw.cuh"

__global__
void decoding_kernel_mw
(
	bool*             d_sig_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	int               level,
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

	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

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
	
	PlanarCoefficients planar_coeffs;
	ScaleChildrenMW    children;
	SubDetailMW        subdetails;

	// Decoding eta
	planar_coeffs =
	{
		d_scale_coeffs.eta0[parent_idx],
		d_scale_coeffs.eta1x[parent_idx],
		d_scale_coeffs.eta1y[parent_idx],
	};

	subdetails = load_subdetails_mw
	(
		d_details.eta0,
		d_details.eta1x,
		d_details.eta1y,
		parent_idx
	);

	children = decode_scale_coeffs(planar_coeffs, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.eta0,
		d_scale_coeffs.eta1x,
		d_scale_coeffs.eta1y,
		child_idx
	);

	// Decoding qx
	planar_coeffs =
	{
		d_scale_coeffs.qx0[parent_idx],
		d_scale_coeffs.qx1x[parent_idx],
		d_scale_coeffs.qx1y[parent_idx],
	};

	subdetails = load_subdetails_mw
	(
		d_details.qx0,
		d_details.qx1x,
		d_details.qx1y,
		parent_idx
	);

	children = decode_scale_coeffs(planar_coeffs, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.qx0,
		d_scale_coeffs.qx1x,
		d_scale_coeffs.qx1y,
		child_idx
	);

	// Decoding qy
	planar_coeffs =
	{
		d_scale_coeffs.qy0[parent_idx],
		d_scale_coeffs.qy1x[parent_idx],
		d_scale_coeffs.qy1y[parent_idx],
	};

	subdetails = load_subdetails_mw
	(
		d_details.qy0,
		d_details.qy1x,
		d_details.qy1y,
		parent_idx
	);

	children = decode_scale_coeffs(planar_coeffs, subdetails);

	store_children_vector
	(
		children,
		d_scale_coeffs.qy0,
		d_scale_coeffs.qy1x,
		d_scale_coeffs.qy1y,
		child_idx
	);
}