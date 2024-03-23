#pragma once

#if _USE_DOUBLES == 1

#define real double
#define real4 double4
#define C(x) x
#define NUM_FRMT "lf"
#define NUM_FIG  ".15"

#else

#define real float
#define real4 float4
#define C(x) x##f
#define NUM_FRMT "f"
#define NUM_FIG  ".8"

#endif