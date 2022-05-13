#include "encode_scale_coefficients.cuh"

__device__
ParentScaleCoefficient encode_scale_coefficients(ChildScaleCoefficients child_scale_coeffs)
{
	return
	{
		encode_scale(child_scale_coeffs.eta),
		encode_scale(child_scale_coeffs.qx),
		encode_scale(child_scale_coeffs.qy),
		encode_scale(child_scale_coeffs.z)
	};
}