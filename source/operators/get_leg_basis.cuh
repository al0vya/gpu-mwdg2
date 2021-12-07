#include "cuda_runtime.h"

#include "LegendreBasis.h"

__device__ __forceinline__
LegendreBasis get_leg_basis
(
	const real& x_face_unit,
	const real& y_face_unit
)
{
	return
	{
		C(1.0),
		sqrt( C(3.0) ) * ( C(2.0) * x_face_unit - C(1.0) ),
		sqrt( C(3.0) ) * ( C(2.0) * y_face_unit - C(1.0) )
	};
}