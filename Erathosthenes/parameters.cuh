#pragma once
#include "cuda_runtime.h"
#include <vector>

struct Dimensions
{
	dim3 gridDim, blockDim, blockIdx, threadIdx;
};
#define PACK_DIMENSIONS() Dimensions{gridDim, blockDim, blockIdx, threadIdx}
#define UNPACK_DIMENSIONS(dimensions)				\
	const dim3 gridDim = dimensions.gridDim;		\
	const dim3 blockDim = dimensions.blockDim;		\
	const dim3 blockIdx = dimensions.blockIdx;		\
	const dim3 threadIdx = dimensions.threadIdx

template<typename T>
struct Array
{
	T* ptr;
	size_t size;
	static Array from_vector(std::vector<T>& vec) { return { vec.data(), vec.size() }; }
	__host__ __device__ T& operator[](size_t i) { return ptr[i]; }
	__host__ __device__ const T& operator[](size_t i) const { return ptr[i]; }
};