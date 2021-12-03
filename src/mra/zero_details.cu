#include "zero_details.cuh"

__global__
void zero_details
(
	Details          d_details,
	real*            d_norm_details,
	int              num_details,
	SolverParameters solver_params
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_details) return;

	d_details.eta0.alpha[idx] = C(0.0);
	d_details.eta0.beta[idx]  = C(0.0);
	d_details.eta0.gamma[idx] = C(0.0);

	d_details.qx0.alpha[idx]  = C(0.0);
	d_details.qx0.beta[idx]   = C(0.0);
	d_details.qx0.gamma[idx]  = C(0.0);

	d_details.qy0.alpha[idx]  = C(0.0);
	d_details.qy0.beta[idx]   = C(0.0);
	d_details.qy0.gamma[idx]  = C(0.0);

	if (solver_params.solver_type == MWDG2)
	{
		d_details.eta1x.alpha[idx] = C(0.0);
		d_details.eta1x.beta[idx]  = C(0.0);
		d_details.eta1x.gamma[idx] = C(0.0);

		d_details.qx1x.alpha[idx] = C(0.0);
		d_details.qx1x.beta[idx]  = C(0.0);
		d_details.qx1x.gamma[idx] = C(0.0);

		d_details.qy1x.alpha[idx] = C(0.0);
		d_details.qy1x.beta[idx]  = C(0.0);
		d_details.qy1x.gamma[idx] = C(0.0);

		d_details.eta1y.alpha[idx] = C(0.0);
		d_details.eta1y.beta[idx]  = C(0.0);
		d_details.eta1y.gamma[idx] = C(0.0);

		d_details.qx1y.alpha[idx] = C(0.0);
		d_details.qx1y.beta[idx]  = C(0.0);
		d_details.qx1y.gamma[idx] = C(0.0);

		d_details.qy1y.alpha[idx] = C(0.0);
		d_details.qy1y.beta[idx]  = C(0.0);
		d_details.qy1y.gamma[idx] = C(0.0);
	}

	d_norm_details[idx] = C(0.0);
}
