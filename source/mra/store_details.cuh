#pragma once

#include "cuda_runtime.h"

#include "../classes/Detail.h"
#include "../classes/Details.h"

__device__ __forceinline__
void store_details
(
	const DetailHW& detail, 
	const Details&  d_details, 
	const int&      h_idx
)
{
	d_details.eta0.alpha[h_idx] = detail.eta.alpha;
	d_details.eta0.beta[h_idx]  = detail.eta.beta;
	d_details.eta0.gamma[h_idx] = detail.eta.gamma;

	d_details.qx0.alpha[h_idx]  = detail.qx.alpha;
	d_details.qx0.beta[h_idx]   = detail.qx.beta;
	d_details.qx0.gamma[h_idx]  = detail.qx.gamma;
	
	d_details.qy0.alpha[h_idx]  = detail.qy.alpha;
	d_details.qy0.beta[h_idx]   = detail.qy.beta;
	d_details.qy0.gamma[h_idx]  = detail.qy.gamma;
	
}

__device__ __forceinline__
void store_details
(
	const DetailMW& detail, 
	const Details&  d_details, 
	const int&      h_idx
)
{
	d_details.eta0.alpha[h_idx] = detail.eta._0.alpha;
	d_details.eta0.beta[h_idx]  = detail.eta._0.beta;
	d_details.eta0.gamma[h_idx] = detail.eta._0.gamma;
	
	d_details.eta1x.alpha[h_idx] = detail.eta._1x.alpha;
	d_details.eta1x.beta[h_idx]  = detail.eta._1x.beta;
	d_details.eta1x.gamma[h_idx] = detail.eta._1x.gamma;
	
	d_details.eta1y.alpha[h_idx] = detail.eta._1y.alpha;
	d_details.eta1y.beta[h_idx]  = detail.eta._1y.beta;
	d_details.eta1y.gamma[h_idx] = detail.eta._1y.gamma;

	d_details.qx0.alpha[h_idx] = detail.qx._0.alpha;
	d_details.qx0.beta[h_idx]  = detail.qx._0.beta;
	d_details.qx0.gamma[h_idx] = detail.qx._0.gamma;
	
	d_details.qx1x.alpha[h_idx] = detail.qx._1x.alpha;
	d_details.qx1x.beta[h_idx]  = detail.qx._1x.beta;
	d_details.qx1x.gamma[h_idx] = detail.qx._1x.gamma;
	
	d_details.qx1y.alpha[h_idx] = detail.qx._1y.alpha;
	d_details.qx1y.beta[h_idx]  = detail.qx._1y.beta;
	d_details.qx1y.gamma[h_idx] = detail.qx._1y.gamma;

	d_details.qy0.alpha[h_idx] = detail.qy._0.alpha;
	d_details.qy0.beta[h_idx]  = detail.qy._0.beta;
	d_details.qy0.gamma[h_idx] = detail.qy._0.gamma;
	
	d_details.qy1x.alpha[h_idx] = detail.qy._1x.alpha;
	d_details.qy1x.beta[h_idx]  = detail.qy._1x.beta;
	d_details.qy1x.gamma[h_idx] = detail.qy._1x.gamma;

	d_details.qy1y.alpha[h_idx] = detail.qy._1y.alpha;
	d_details.qy1y.beta[h_idx]  = detail.qy._1y.beta;
	d_details.qy1y.gamma[h_idx] = detail.qy._1y.gamma;
	/*
	d_details.z0.alpha[h_idx] = detail.z._0.alpha;
	d_details.z0.beta[h_idx]  = detail.z._0.beta;
	d_details.z0.gamma[h_idx] = detail.z._0.gamma;
	
	d_details.z1x.alpha[h_idx] = detail.z._1x.alpha;
	d_details.z1x.beta[h_idx]  = detail.z._1x.beta;
	d_details.z1x.gamma[h_idx] = detail.z._1x.gamma;

	d_details.z1y.alpha[h_idx] = detail.z._1y.alpha;
	d_details.z1y.beta[h_idx]  = detail.z._1y.beta;
	d_details.z1y.gamma[h_idx] = detail.z._1y.gamma;
	*/
}