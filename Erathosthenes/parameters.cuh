#pragma once
#include "cuda_runtime.h"
#include <vector>
#include "interlocked.cuh"

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

template<typename T, typename Access = T&>
struct Array
{
	T* ptr;
	size_t size;
	size_t memory;
	static Array from_vector(std::vector<T>& vec) { return { vec.data(), vec.size(), vec.size() * sizeof(T) }; }
	static Array from_carray(T* ptr, size_t size) { return { ptr, size, size * sizeof(T) }; }
	static Array from_malloc(void* ptr, size_t memory) { return { reinterpret_cast<T*>(ptr), memory / sizeof(T), memory }; }
	Array subarray(size_t offset, size_t _size) { return { ptr + offset, _size, _size * sizeof(T) }; }
	const Array subarray(size_t offset, size_t _size) const { return { ptr + offset, _size, _size * sizeof(T) }; }
	__host__ __device__ Access operator[](size_t i) { return ptr[i]; }
	__host__ __device__ const Access operator[](size_t i) const { return ptr[i]; }
	constexpr static size_t value_per_byte() { return 0ull; }
};

struct Bits
{
	long data;
	struct BitAccess
	{
		long* data;
		size_t offset;
		operator bool() const { return static_cast<bool>(*data & 1u << offset); }
		__host__ __device__ void operator=(bool v)
		{
			if (v) 
				interlocked_or(data, 1u << offset);
			else 
				interlocked_and(data, ~(1u << offset));
		}
	};
};

template<>
constexpr size_t Array<Bits, Bits::BitAccess>::value_per_byte() { return 8ull; }
template<>
constexpr size_t Array<bool>::value_per_byte() { return 1ull; }

template<>
Array<Bits, Bits::BitAccess> Array<Bits, Bits::BitAccess>::from_malloc(void* ptr, size_t memory) 
{ 
	return { reinterpret_cast<Bits*>(ptr), static_cast<size_t>(memory * Array<Bits, Bits::BitAccess>::value_per_byte()), memory }; 
}
template<>
Array<Bits, Bits::BitAccess> Array<Bits, Bits::BitAccess>::subarray(size_t offset, size_t _size) 
{ 
	return { reinterpret_cast<Bits*>(reinterpret_cast<bool*>(ptr) + offset / Array<Bits, Bits::BitAccess>::value_per_byte()), _size, _size / Array<Bits, Bits::BitAccess>::value_per_byte() };
}
template<>
const Array<Bits, Bits::BitAccess> Array<Bits, Bits::BitAccess>::subarray(size_t offset, size_t _size) const 
{
	return { reinterpret_cast<Bits*>(reinterpret_cast<bool*>(ptr) + offset / Array<Bits, Bits::BitAccess>::value_per_byte()), _size, _size / Array<Bits, Bits::BitAccess>::value_per_byte() };
}

static_assert(sizeof(Bits) == sizeof(uint32_t), "Bits needs to be 32 bits!!!");
template<>
__host__ __device__ Bits::BitAccess Array<Bits, Bits::BitAccess>::operator[](size_t i) { return { &ptr[i / 32ull].data, i % 32ull }; }
template<>
__host__ __device__ const Bits::BitAccess Array<Bits, Bits::BitAccess>::operator[](size_t i) const { return { &ptr[i / 32ull].data, i % 32ull }; }

#if USE_BITS
using ResultArray = Array<Bits, Bits::BitAccess>;
#else
using ResultArray = Array<bool>;
#endif