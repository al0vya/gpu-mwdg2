#pragma once

#include "cuda_runtime.h"

#include "FlowCoeffs.h"

__device__ __forceinline__
real get_bed_src_x
(
	const real& z_w_intermediate,
	const real& z_e_intermediate,
	const real& h_star_w_pos,
	const real& h_star_e_neg,
	const real& eta_w_pos,
	const real& eta_e_neg,
	const real& g,
	const real& dx_loc,
	int idx
)
{
	real z_star_w = z_w_intermediate - max( C(0.0), -(eta_w_pos - z_w_intermediate) );
	real z_star_e = z_e_intermediate - max( C(0.0), -(eta_e_neg - z_e_intermediate) );

	return g * C(0.5) * (h_star_w_pos + h_star_e_neg) * (z_star_e - z_star_w) / dx_loc;
}

__device__ __forceinline__
real get_bed_src_y
(
	const real& z_s_intermediate,
	const real& z_n_intermediate,
	const real& h_star_s_pos,
	const real& h_star_n_neg,
	const real& eta_s_pos,
	const real& eta_n_neg,
	const real& g,
	const real& dy_loc
)
{
	real z_star_s = z_s_intermediate - max( C(0.0), -(eta_s_pos - z_s_intermediate) );
	real z_star_n = z_n_intermediate - max( C(0.0), -(eta_n_neg - z_n_intermediate) );

	return g * C(0.5) * (h_star_s_pos + h_star_n_neg) * (z_star_n - z_star_s) / dy_loc;
}

__device__ __forceinline__
FlowCoeffs get_bed_src_x
(
	const real& eta_e_neg,
	const real& eta_w_pos,
	const real& z_inter_e,
	const real& z_inter_w,
	const real& h0x_star,
	const real& h1x_star,
	const real& g,
	const real& dx_loc,
	const FlowCoeffs& coeffs,
	int idx
)
{
	real z_star_e = z_inter_e - max( C(0.0), -(eta_e_neg - z_inter_e) );
	real z_star_w = z_inter_w - max( C(0.0), -(eta_w_pos - z_inter_w) );

	real z1x_star = (z_star_e - z_star_w) / (C(2.0) * sqrt(C(3.0)));

	FlowCoeffs Sbx = {};

	Sbx.qx0  = -C(2.0) * sqrt(C(3.0)) * g * h0x_star * z1x_star / dx_loc;
	Sbx.qx1x = -C(2.0) * sqrt(C(3.0)) * g * h1x_star * z1x_star / dx_loc;
	
	if (false)//idx == 65536)
	{
		printf("z_star_e: %f\n", z_star_e);
		printf("z_star_w: %f\n", z_star_w);
		printf("z_inter_e: %f\n", z_inter_e);
		printf("z_inter_w: %f\n", z_inter_w);
		printf("eta_e_neg: %f\n", eta_e_neg);
		printf("eta_w_pos: %f\n", eta_w_pos);
		printf("z1x_star: %f\n", z1x_star);
		printf("h0x_star: %f\n", h0x_star);
		printf("h1x_star: %f\n", h1x_star);
		printf("Sbx.qx0: %f\n", Sbx.qx0);
		printf("Sbx.qx1x: %f\n", Sbx.qx1x);
		printf("\n");
	}
	
	return Sbx;
}

__device__ __forceinline__
FlowCoeffs get_bed_src_y
(
	const real& eta_n_neg,
	const real& eta_s_pos,
	const real& z_inter_n,
	const real& z_inter_s,
	const real& h0y_star,
	const real& h1y_star,
	const real& g,
	const real& dy_loc,
	const FlowCoeffs& coeffs
)
{
	real z_star_n = z_inter_n - max( C(0.0), -(eta_n_neg - z_inter_n) );
	real z_star_s = z_inter_s - max( C(0.0), -(eta_s_pos - z_inter_s) );

	real z1y_star = (z_star_n - z_star_s) / ( C(2.0) * sqrt( C(3.0) ) );

	FlowCoeffs Sby = {};

	Sby.qy0  = -C(2.0) * sqrt( C(3.0) ) * g * h0y_star * z1y_star / dy_loc;
	Sby.qy1y = -C(2.0) * sqrt( C(3.0) ) * g * h1y_star * z1y_star / dy_loc;

	return Sby;
}