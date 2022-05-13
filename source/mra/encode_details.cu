#include "encode_details.cuh"

__device__
DetailHW encode_details(const ChildScaleCoeffsHW& child_coeffs)
{
	SubDetailHW eta =
	{
		encode_detail_alpha(child_coeffs.eta),
		encode_detail_beta(child_coeffs.eta),
		encode_detail_gamma(child_coeffs.eta)
	};

	SubDetailHW qx =
	{
		encode_detail_alpha(child_coeffs.qx),
		encode_detail_beta(child_coeffs.qx),
		encode_detail_gamma(child_coeffs.qx)
	};

	SubDetailHW qy =
	{
		encode_detail_alpha(child_coeffs.qy),
		encode_detail_beta(child_coeffs.qy),
		encode_detail_gamma(child_coeffs.qy)
	};
	
	SubDetailHW z =
	{
		encode_detail_alpha(child_coeffs.z),
		encode_detail_beta(child_coeffs.z),
		encode_detail_gamma(child_coeffs.z)
	};

	return
	{
		eta,
		qx,
		qy,
		z
	};
}

__device__
real encode_detail_alpha(const ScaleChildrenHW& u)
{
	return C(0.5) * ( H0 * (G0 * u.child_0 + G1 * u.child_2) + H1 * (G0 * u.child_1 + G1 * u.child_3) );
}

__device__
real encode_detail_beta(const ScaleChildrenHW& u)
{
	return C(0.5) * ( G0 * (H0 * u.child_0 + H1 * u.child_2) + G1 * (H0 * u.child_1 + H1 * u.child_3) );
}

__device__
real encode_detail_gamma(const ScaleChildrenHW& u)
{
	return C(0.5) * ( G0 * (G0 * u.child_0 + G1 * u.child_2) + G1 * (G0 * u.child_1 + G1 * u.child_3) );
}

__device__
real encode_detail_alpha_0(const ScaleChildrenMW& u)
{
	return (GA0_11 * u._0.child_0 + GA0_12 * u._1x.child_0 + GA0_13 * u._1y.child_0 +
		    GA1_11 * u._0.child_2 + GA1_12 * u._1x.child_2 + GA1_13 * u._1y.child_2 +
		    GA2_11 * u._0.child_1 + GA2_12 * u._1x.child_1 + GA2_13 * u._1y.child_1 +
		    GA3_11 * u._0.child_3 + GA3_12 * u._1x.child_3 + GA3_13 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_beta_0(const ScaleChildrenMW& u)
{
	return (GB0_11 * u._0.child_0 + GB0_12 * u._1x.child_0 + GB0_13 * u._1y.child_0 +
		    GB1_11 * u._0.child_2 + GB1_12 * u._1x.child_2 + GB1_13 * u._1y.child_2 +
		    GB2_11 * u._0.child_1 + GB2_12 * u._1x.child_1 + GB2_13 * u._1y.child_1 +
		    GB3_11 * u._0.child_3 + GB3_12 * u._1x.child_3 + GB3_13 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_gamma_0(const ScaleChildrenMW& u)
{
	return (GC0_11 * u._0.child_0 + GC0_12 * u._1x.child_0 + GC0_13 * u._1y.child_0 +
		    GC1_11 * u._0.child_2 + GC1_12 * u._1x.child_2 + GC1_13 * u._1y.child_2 +
		    GC2_11 * u._0.child_1 + GC2_12 * u._1x.child_1 + GC2_13 * u._1y.child_1 +
		    GC3_11 * u._0.child_3 + GC3_12 * u._1x.child_3 + GC3_13 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_alpha_1x(const ScaleChildrenMW& u)
{
	return (GA0_21 * u._0.child_0 + GA0_22 * u._1x.child_0 + GA0_23 * u._1y.child_0 +
		    GA1_21 * u._0.child_2 + GA1_22 * u._1x.child_2 + GA1_23 * u._1y.child_2 +
		    GA2_21 * u._0.child_1 + GA2_22 * u._1x.child_1 + GA2_23 * u._1y.child_1 +
		    GA3_21 * u._0.child_3 + GA3_22 * u._1x.child_3 + GA3_23 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_beta_1x(const ScaleChildrenMW& u)
{
	return (GB0_21 * u._0.child_0 + GB0_22 * u._1x.child_0 + GB0_23 * u._1y.child_0 +
		    GB1_21 * u._0.child_2 + GB1_22 * u._1x.child_2 + GB1_23 * u._1y.child_2 +
		    GB2_21 * u._0.child_1 + GB2_22 * u._1x.child_1 + GB2_23 * u._1y.child_1 +
		    GB3_21 * u._0.child_3 + GB3_22 * u._1x.child_3 + GB3_23 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_gamma_1x(const ScaleChildrenMW& u)

{
	return (GC0_21 * u._0.child_0 + GC0_22 * u._1x.child_0 + GC0_23 * u._1y.child_0 +
		    GC1_21 * u._0.child_2 + GC1_22 * u._1x.child_2 + GC1_23 * u._1y.child_2 +
		    GC2_21 * u._0.child_1 + GC2_22 * u._1x.child_1 + GC2_23 * u._1y.child_1 +
		    GC3_21 * u._0.child_3 + GC3_22 * u._1x.child_3 + GC3_23 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_alpha_1y(const ScaleChildrenMW& u)
{
	return (GA0_31 * u._0.child_0 + GA0_32 * u._1x.child_0 + GA0_33 * u._1y.child_0 +
		    GA1_31 * u._0.child_2 + GA1_32 * u._1x.child_2 + GA1_33 * u._1y.child_2 +
		    GA2_31 * u._0.child_1 + GA2_32 * u._1x.child_1 + GA2_33 * u._1y.child_1 +
		    GA3_31 * u._0.child_3 + GA3_32 * u._1x.child_3 + GA3_33 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_beta_1y(const ScaleChildrenMW& u)
{
	return (GB0_31 * u._0.child_0 + GB0_32 * u._1x.child_0 + GB0_33 * u._1y.child_0 +
		    GB1_31 * u._0.child_2 + GB1_32 * u._1x.child_2 + GB1_33 * u._1y.child_2 +
		    GB2_31 * u._0.child_1 + GB2_32 * u._1x.child_1 + GB2_33 * u._1y.child_1 +
		    GB3_31 * u._0.child_3 + GB3_32 * u._1x.child_3 + GB3_33 * u._1y.child_3) / C(2.0);
}

__device__
real encode_detail_gamma_1y(const ScaleChildrenMW& u)
{
	return (GC0_31 * u._0.child_0 + GC0_32 * u._1x.child_0 + GC0_33 * u._1y.child_0 +
		    GC1_31 * u._0.child_2 + GC1_32 * u._1x.child_2 + GC1_33 * u._1y.child_2 +
		    GC2_31 * u._0.child_1 + GC2_32 * u._1x.child_1 + GC2_33 * u._1y.child_1 +
		    GC3_31 * u._0.child_3 + GC3_32 * u._1x.child_3 + GC3_33 * u._1y.child_3) / C(2.0);
}

__device__
SubDetailHW encode_detail_0(const ScaleChildrenMW& u)
{
	return
	{
		encode_detail_alpha_0(u),
		encode_detail_beta_0(u),
		encode_detail_gamma_0(u)
	};
}

__device__
SubDetailHW encode_detail_1x(const ScaleChildrenMW& u)
{
	return
	{
		encode_detail_alpha_1x(u),
		encode_detail_beta_1x(u),
		encode_detail_gamma_1x(u)
	};
}

__device__
SubDetailHW encode_detail_1y(const ScaleChildrenMW& u)
{
	return
	{
		encode_detail_alpha_1y(u),
		encode_detail_beta_1y(u),
		encode_detail_gamma_1y(u)
	};
}

__device__
SubDetailMW encode_detail(const ScaleChildrenMW& u)
{
	return
	{
		encode_detail_0 (u),
		encode_detail_1x(u),
		encode_detail_1y(u)
	};
}

__device__
DetailMW encode_details(const ChildScaleCoeffsMW& child_coeffs)
{
	return
	{
		encode_detail(child_coeffs.eta),
		encode_detail(child_coeffs.qx),
		encode_detail(child_coeffs.qy),
		encode_detail(child_coeffs.z)
	};
}