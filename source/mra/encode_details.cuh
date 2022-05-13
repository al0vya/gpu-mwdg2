#pragma once

#include "cuda_runtime.h"

#include "Detail.h"
#include "ChildScaleCoeffs.h"
#include "Filters.h"

// encodes the details alpha, beta and gamma for eta, qx, qy and z
__device__
real encode_detail_alpha(const ScaleChildrenHW& u);

__device__
real encode_detail_beta(const ScaleChildrenHW& u);

__device__
real encode_detail_gamma(const ScaleChildrenHW& u);

__device__
DetailHW encode_details(const ChildScaleCoeffsHW& child_coeffs);

__device__
real encode_detail_alpha(const ScaleChildrenHW& u);

__device__
real encode_detail_beta(const ScaleChildrenHW& u);

__device__
real encode_detail_gamma(const ScaleChildrenHW& u);

__device__
real encode_detail_alpha_0(const ScaleChildrenMW& u);

__device__
real encode_detail_beta_0(const ScaleChildrenMW& u);

__device__
real encode_detail_gamma_0(const ScaleChildrenMW& u);

__device__
real encode_detail_alpha_1x(const ScaleChildrenMW& u);

__device__
real encode_detail_beta_1x(const ScaleChildrenMW& u);

__device__
real encode_detail_gamma_1x(const ScaleChildrenMW& u);

__device__
real encode_detail_alpha_1y(const ScaleChildrenMW& u);

__device__
real encode_detail_beta_1y(const ScaleChildrenMW& u);

__device__
real encode_detail_gamma_1y(const ScaleChildrenMW& u);

__device__
SubDetailHW encode_detail_0(const ScaleChildrenMW& u);

__device__
SubDetailHW encode_detail_1x(const ScaleChildrenMW& u);

__device__
SubDetailHW encode_detail_1y(const ScaleChildrenMW& u);

__device__
SubDetailMW encode_detail(const ScaleChildrenMW& u);

__device__
DetailMW encode_details(const ChildScaleCoeffsMW& child_coeffs);