#include "load_parent_scale_coefficients.cuh"

__device__
ParentScaleCoeffsHW load_parent_scale_coefficients
(
	ScaleCoefficients& d_scale_coeffs,
	int&               h_idx
)
{
	return
	{
		d_scale_coeffs.eta0[h_idx],
		d_scale_coeffs.qx0[h_idx],
		d_scale_coeffs.qy0[h_idx],
		d_scale_coeffs.z0[h_idx]
	};
}