#pragma once

#include "../utilities/cuda_utils.cuh"

#if _USE_TRACER == 1

#include "nvtx3/nvToolsExt.h"

typedef struct Tracer
{
    Tracer(const char* label)
    {
        nvtxRangePushA(label);
    }
    
    ~Tracer()
    {
        CHECK_CUDA_ERROR( peek() );
        CHECK_CUDA_ERROR( sync() );

        nvtxRangePop();
    }

} Tracer;

#define TRACE(name) Tracer tracer(name);

#else

#define TRACE(name)

#endif
