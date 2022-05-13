#pragma once

#include "cuda_runtime.h"

#include "Filters.h"
#include "SubDetail.h"
#include "ScaleChildren.h"

__device__
real decode_0(real& u, SubDetailHW& sub_detail);

__device__
real decode_1(real& u, SubDetailHW& sub_detail);

__device__
real decode_2(real& u, SubDetailHW& sub_detail);

__device__
real decode_3(real& u, SubDetailHW& sub_detail);

__device__
ScaleChildrenHW decode_scale_children
(
	real      u,
	SubDetailHW sub_detail
);

__device__
real decode_0(real& u, SubDetailHW& sub_detail);

__device__
real decode_2(real& u, SubDetailHW& sub_detail);

__device__
real decode_1(real& u, SubDetailHW& sub_detail);

__device__
real decode_3(real& u, SubDetailHW& sub_detail);

__device__
real decode_0_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_0_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_0_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_2_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_2_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_2_1y
(
	const real& u0,	 const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_1_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_1_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,	
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_1_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_3_0
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_3_1x
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);

__device__
real decode_3_1y
(
	const real& u0,  const real& u1x,  const real& u1y,
	const real& da0, const real& da1x, const real& da1y,
	const real& db0, const real& db1x, const real& db1y,
	const real& dg0, const real& dg1x, const real& dg1y
);