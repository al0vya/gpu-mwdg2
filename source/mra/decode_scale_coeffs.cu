#include "decode_scale_coeffs.cuh"

__device__
ChildScaleCoeffsHW decode_scale_coeffs
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

__device__
ChildScaleCoeffsMW decode_scale_coeffs
(
	ParentScaleCoeffsMW& u,
	DetailMW&            d
)
{
	return
	{
		{
			{
				decode_0_0(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_1_0(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_2_0(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_3_0(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma)
			},
			{
				decode_0_1x(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_1_1x(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_2_1x(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_3_1x(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma)
			},
			{
				decode_0_1y(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_1_1y(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_2_1y(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma),
				decode_3_1y(u._0.eta, u._1x.eta, u._1y.eta, d.eta._0.alpha, d.eta._1x.alpha, d.eta._1y.alpha, d.eta._0.beta, d.eta._1x.beta, d.eta._1y.beta, d.eta._0.gamma, d.eta._1x.gamma, d.eta._1y.gamma)
			}
		},
		{
			{
				decode_0_0(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_1_0(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_2_0(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_3_0(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma)
			},
			{
				decode_0_1x(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_1_1x(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_2_1x(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_3_1x(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma)
			},
			{
				decode_0_1y(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_1_1y(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_2_1y(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma),
				decode_3_1y(u._0.qx, u._1x.qx, u._1y.qx, d.qx._0.alpha, d.qx._1x.alpha, d.qx._1y.alpha, d.qx._0.beta, d.qx._1x.beta, d.qx._1y.beta, d.qx._0.gamma, d.qx._1x.gamma, d.qx._1y.gamma)
			}
		},
		{
			{
				decode_0_0(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_1_0(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_2_0(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_3_0(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma)
			},
			{
				decode_0_1x(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_1_1x(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_2_1x(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_3_1x(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma)
			},
			{
				decode_0_1y(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_1_1y(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_2_1y(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma),
				decode_3_1y(u._0.qy, u._1x.qy, u._1y.qy, d.qy._0.alpha, d.qy._1x.alpha, d.qy._1y.alpha, d.qy._0.beta, d.qy._1x.beta, d.qy._1y.beta, d.qy._0.gamma, d.qy._1x.gamma, d.qy._1y.gamma)
			}
		},
		{
			{
				C(0.0), C(0.0), C(0.0), C(0.0)
			},
			{
				C(0.0), C(0.0), C(0.0), C(0.0)
			},
			{
				C(0.0), C(0.0), C(0.0), C(0.0)
			}
		}
	};
}