#pragma once

#include "cuda_runtime.h"

#include "FlowCoeffs.h"

__device__
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
);

__device__
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
);

__device__
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
);

__device__
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
);