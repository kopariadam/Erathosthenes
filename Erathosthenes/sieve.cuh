#pragma once
#include "hardware_defines.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "threadable_function.h"
#include "parameters.cuh"
#include <chrono>

struct SieveParams
{
	ResultArray result;
	size_t offset;
	const Array<uint32_t> knownPrimes;
	uint32_t startingPrimeIdx;
};

__host__ __device__ void sieve_impl(const Dimensions& dimensions, SieveParams& params)
{
	UNPACK_DIMENSIONS(dimensions);
	auto primeIdx = params.startingPrimeIdx + blockDim.x * threadIdx.y + threadIdx.x;
	if (primeIdx >= params.knownPrimes.size) return;
	size_t step = params.knownPrimes[primeIdx];
	if (step == 2ull) return;

	size_t numberIdx = blockIdx.x;
	auto numberBlockSize = 2ull * params.result.size / gridDim.x;
	auto startingNumber = numberBlockSize * numberIdx + params.offset;
	if (startingNumber > step)
		startingNumber = (((startingNumber - 1ull) / step + 1ull) | 1ull) * step;
	startingNumber = max(startingNumber, step * step);
	
	auto startingIndex = (startingNumber - params.offset) / 2ull;
	auto endingIndex = numberBlockSize * (numberIdx + 1ull) / 2ull;

	for (auto i = startingIndex; i < endingIndex; i += step)
		params.result[i] = false;
}

cudaError_t cudaStatus;
#define CUDA_STATUS_CHECK(A) do {cudaStatus = (A); if(cudaStatus!=cudaSuccess) {std::cout<<"cuda error "<<__LINE__<<" "<<__FILE__<<"\n";}} while(0)

__global__ void sieve_cuda(SieveParams params)
{
	sieve_impl(PACK_DIMENSIONS(), params);
}

THREADABLE_FUNCTION_START(sieve_host, SieveParams params)
	sieve_impl(PACK_DIMENSIONS(), params);
THREADABLE_FUNCTION_END

template<bool gpu_enabled>
void sieve(ResultArray result, size_t offset, Array<uint32_t>& known_primes)
{
	auto startTime = std::chrono::high_resolution_clock::now();

	if (gpu_enabled)
	{
		compute_log << "Copying arrays to GPU\n";
		cudaSetDevice(0);

		uint32_t* cudaKnownPrimes;
		CUDA_STATUS_CHECK(cudaMalloc(&cudaKnownPrimes, known_primes.memory));
		CUDA_STATUS_CHECK(cudaMemcpy(cudaKnownPrimes, known_primes.ptr, known_primes.memory, cudaMemcpyHostToDevice));

		void* cudaResult;
		CUDA_STATUS_CHECK(cudaMalloc(&cudaResult, result.memory));
		CUDA_STATUS_CHECK(cudaMemset(cudaResult, 0xff, result.memory));
		compute_log << "Arrays copied to GPU\n";

		for (auto i = 0u; i < known_primes.size; i += BLOCK_SIZE * BLOCK_SIZE)
		{
			auto grid = dim3(BLOCK_COUNT);
			auto block = dim3(BLOCK_SIZE, BLOCK_SIZE);
			SieveParams params{ ResultArray::from_malloc(cudaResult, result.memory), offset, Array<uint32_t>::from_carray(cudaKnownPrimes, known_primes.size), i };
			sieve_cuda <<< grid, block >>>(params);
			CUDA_STATUS_CHECK(cudaGetLastError());
		}
		CUDA_STATUS_CHECK(cudaDeviceSynchronize());
		compute_log << "Calculation done\n";

		compute_log << "Copying back result\n";
		CUDA_STATUS_CHECK(cudaMemcpy(result.ptr, cudaResult, result.memory, cudaMemcpyDeviceToHost));
		compute_log << "Copying back done\n";

		CUDA_STATUS_CHECK(cudaFree(cudaKnownPrimes));
		CUDA_STATUS_CHECK(cudaFree(cudaResult));
		cudaDeviceReset();
	}
	else
	{
		memset(result.ptr, 0xff, result.memory);
		for (auto i = 0u; i < known_primes.size; i += THREAD_COUNT)
		{
			compute_log << "Calculating " << i << "\n";
			SieveParams params{ result, offset, known_primes, i };
			THREADABLE_CALL(THREAD_COUNT, sieve_host, params); //BLOCK_CLOUNT = 1
		}
		compute_log << "Calculation done" << "\n";
	}
	if (offset == 0ull)
		result[0] = false;

	auto endTime = std::chrono::high_resolution_clock::now();
	auto calculationTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
	compute_log << "Calculation took " << calculationTime << "ms\n";
}