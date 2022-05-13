#include "decode_scale_coefficients.cuh"

__device__
ChildScaleCoeffsHW decode_scale_coefficients
(
	ParentScaleCoeffsHW& parent_coeffs,
	DetailHW&            detail
)
{
	return
	{
		decode_scale_children(parent_coeffs.eta, detail.eta),
		decode_scale_children(parent_coeffs.qx,  detail.qx),
		decode_scale_children(parent_coeffs.qy,  detail.qy),
		{ 0, 0, 0, 0 }
	};
}