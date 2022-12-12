#pragma once

#include "../classes/SubDetail.h"
#include "../classes/Maxes.h"

typedef struct DetailHW
{
	SubDetailHW eta;
	SubDetailHW qx;
	SubDetailHW qy;
	SubDetailHW z;

	__device__ __forceinline__
	real get_norm_detail(Maxes maxes)
	{
		real norm_detail = C(0.0);

		real eta_norm = eta.get_max() / maxes.eta;
		real qx_norm  = qx.get_max()  / maxes.qx;
		real qy_norm  = qy.get_max()  / maxes.qy;
		real z_norm   = z.get_max()   / maxes.z;

		norm_detail = max(eta_norm, qx_norm);
		norm_detail = max(qy_norm,  norm_detail);
		norm_detail = max(z_norm,   norm_detail);

		return norm_detail;
	}

} DetailHW;

typedef struct DetailMW
{
	SubDetailMW eta;
	SubDetailMW qx;
	SubDetailMW qy;
	SubDetailMW z;

	__device__ __forceinline__
	real get_norm_detail(Maxes maxes)
	{
		real norm_detail = C(0.0);

		real eta_norm = eta.get_max() / maxes.eta;
		real qx_norm  =  qx.get_max() / maxes.qx;
		real qy_norm  =  qy.get_max() / maxes.qy;
		real z_norm   =   z.get_max() / maxes.z;

		norm_detail = max(eta_norm, qx_norm);
		norm_detail = max(qy_norm,  norm_detail);
		norm_detail = max(z_norm,   norm_detail);

		return norm_detail;
	}

} DetailMW;