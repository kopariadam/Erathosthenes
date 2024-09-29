#pragma once
#include "cuda_runtime.h"
#include <atomic>

__host__ __device__ void interlocked_and(long* data, long v)
{
#ifdef __CUDA_ARCH__
	atomicAnd(reinterpret_cast<int*>(data), v);
#else
	_InterlockedAnd(data, v);
#endif
}
__host__ __device__ void interlocked_or(long* data, long v)
{
#ifdef __CUDA_ARCH__
	atomicOr(reinterpret_cast<int*>(data), v);
#else
	_InterlockedOr(data, v);
#endif
}