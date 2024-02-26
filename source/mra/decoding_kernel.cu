#include "decoding_kernel.cuh"

__global__
void decoding_kernel
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
		real     coeffs[4 * THREADS_PER_BLOCK];

	} shared;

	__shared__ HierarchyIndex shared_parents[THREADS_PER_BLOCK];

	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	HierarchyIndex parent = curr_lvl_idx + idx;

	int is_sig = d_sig_details[parent];

	int thread_prefix_sum = 0;

	int num_sig_details = 0;

	block_scan(shared.temp_storage).ExclusiveSum
	(
		is_sig,
		thread_prefix_sum,
		num_sig_details
	);

	__syncthreads();

	if (is_sig) shared_parents[thread_prefix_sum] = parent;

	__syncthreads();

	if (t_idx >= num_sig_details) return;

	parent = shared_parents[t_idx];

	if (solver_params.solver_type == HWFV1)
	{
		ParentScaleCoeffsHW parents = load_parent_scale_coeffs_hw(d_scale_coeffs, parent);
		DetailHW            detail = load_details_hw(d_details, parent);
		ChildScaleCoeffsHW  children = decode_scale_coeffs(parents, detail);

		// storing eta
		shared.coeffs[4 * t_idx + 0] = children.eta.child_0;
		shared.coeffs[4 * t_idx + 1] = children.eta.child_1;
		shared.coeffs[4 * t_idx + 2] = children.eta.child_2;
		shared.coeffs[4 * t_idx + 3] = children.eta.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.eta0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qx
		shared.coeffs[4 * t_idx + 0] = children.qx.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qx.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qx.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qx.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qx0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qy
		shared.coeffs[4 * t_idx + 0] = children.qy.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qy.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qy.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qy.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qy0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();
	}
	else if (solver_params.solver_type == MWDG2)
	{
		ParentScaleCoeffsMW parents = load_parent_scale_coeffs_mw(d_scale_coeffs, parent);
		DetailMW            detail = load_details_mw(d_details, parent);
		ChildScaleCoeffsMW  children = decode_scale_coeffs(parents, detail);

		// storing eta
		shared.coeffs[4 * t_idx + 0] = children.eta._0.child_0;
		shared.coeffs[4 * t_idx + 1] = children.eta._0.child_1;
		shared.coeffs[4 * t_idx + 2] = children.eta._0.child_2;
		shared.coeffs[4 * t_idx + 3] = children.eta._0.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.eta0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qx
		shared.coeffs[4 * t_idx + 0] = children.qx._0.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qx._0.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qx._0.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qx._0.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qx0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qy
		shared.coeffs[4 * t_idx + 0] = children.qy._0.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qy._0.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qy._0.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qy._0.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qy0[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing eta
		shared.coeffs[4 * t_idx + 0] = children.eta._1x.child_0;
		shared.coeffs[4 * t_idx + 1] = children.eta._1x.child_1;
		shared.coeffs[4 * t_idx + 2] = children.eta._1x.child_2;
		shared.coeffs[4 * t_idx + 3] = children.eta._1x.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.eta1x[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qx
		shared.coeffs[4 * t_idx + 0] = children.qx._1x.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qx._1x.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qx._1x.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qx._1x.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qx1x[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qy
		shared.coeffs[4 * t_idx + 0] = children.qy._1x.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qy._1x.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qy._1x.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qy._1x.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qy1x[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing eta
		shared.coeffs[4 * t_idx + 0] = children.eta._1y.child_0;
		shared.coeffs[4 * t_idx + 1] = children.eta._1y.child_1;
		shared.coeffs[4 * t_idx + 2] = children.eta._1y.child_2;
		shared.coeffs[4 * t_idx + 3] = children.eta._1y.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.eta1y[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qx
		shared.coeffs[4 * t_idx + 0] = children.qx._1y.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qx._1y.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qx._1y.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qx._1y.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qx1y[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();

		// storing qy
		shared.coeffs[4 * t_idx + 0] = children.qy._1y.child_0;
		shared.coeffs[4 * t_idx + 1] = children.qy._1y.child_1;
		shared.coeffs[4 * t_idx + 2] = children.qy._1y.child_2;
		shared.coeffs[4 * t_idx + 3] = children.qy._1y.child_3;
		__syncthreads();

		for (int i = 0; i < 4; i++)
		{
			HierarchyIndex shared_idx = t_idx + i * num_sig_details;
			HierarchyIndex child = next_lvl_idx + 4 * (shared_parents[shared_idx / 4] - curr_lvl_idx) + shared_idx % 4;

			d_scale_coeffs.qy1y[child] = shared.coeffs[shared_idx];
		}
		__syncthreads();
	}
}