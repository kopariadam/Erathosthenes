#pragma once
#include "cuda_runtime.h"

#include <thread>
#define THREADABLE_FUNCTION_START(name, ...)																				\
struct ThreadableFunction##name																								\
{																															\
	dim3 gridDim, blockDim, threadIdx, blockIdx;																			\
	ThreadableFunction##name(unsigned int blockDim, unsigned int threadIdx)													\
		: gridDim(1, 1, 1), blockDim(blockDim, 1, 1), blockIdx(0, 0, 0), threadIdx(threadIdx, 0, 0) {}						\
	void operator()( __VA_ARGS__ )																							\
	{


#define THREADABLE_FUNCTION_END																								\
	}																														\
};

#define THREADABLE_CALL(count, name, ...)																					\
do																															\
{																															\
	std::thread threads[count];																								\
	for(int _i = 0; _i < count; _i++)																						\
	{																														\
		threads[_i] = std::thread(ThreadableFunction##name(count , _i), __VA_ARGS__);										\
	}																														\
	for(int _i = 0; _i < count; _i++)																						\
	{																														\
		threads[_i].join();																									\
	}																														\
} while (0)