#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "threadable_function.h"
#include "parameters.cuh"

struct SieveParams
{
	Array<bool> result;
	const Array<size_t> knownPrimes;
	size_t startingPrimeIdx;
};

__host__ __device__ void sieve_impl(const Dimensions& dimensions, SieveParams& params)
{
	UNPACK_DIMENSIONS(dimensions);
	auto primeIdx = params.startingPrimeIdx + (size_t)blockDim.x * (size_t)threadIdx.y + (size_t)threadIdx.x;
	if (primeIdx >= params.knownPrimes.size) return;
	auto step = params.knownPrimes[primeIdx];

	size_t numberIdx = blockIdx.x;
	auto numberBlockSize = params.result.size / gridDim.x;
	auto startingNumber = numberBlockSize * numberIdx;
	if (startingNumber > 0)
		startingNumber = ((startingNumber - 1) / step + 1) * step;
	auto endingNumber = numberBlockSize * (numberIdx + 1);

	for (auto i = startingNumber; i < endingNumber; i += step)
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
void sieve(Array<bool> result, std::vector<size_t>& known_primes)
{
	memset(result.ptr, 1, result.size * sizeof(result[0]));
	result[0] = false;
	result[1] = false;
	if (gpu_enabled)
	{
		constexpr auto BLOCK_COUNT = 1ll << 20;
		constexpr auto BLOCK_SIZE = 16ll;

		std::cout << "Copying arrays to GPU" << std::endl;
		cudaSetDevice(0);

		size_t* cudaKnownPrimes;
		CUDA_STATUS_CHECK(cudaMalloc(&cudaKnownPrimes, known_primes.size() * sizeof(known_primes[0])));
		CUDA_STATUS_CHECK(cudaMemcpy(cudaKnownPrimes, known_primes.data(), known_primes.size() * sizeof(known_primes[0]), cudaMemcpyHostToDevice));

		bool* cudaResult;
		CUDA_STATUS_CHECK(cudaMalloc(&cudaResult, result.size));
		CUDA_STATUS_CHECK(cudaMemcpy(cudaResult, result.ptr, result.size * sizeof(result[0]), cudaMemcpyHostToDevice));
		std::cout << "Arrays copied to GPU" << std::endl;

		for (auto i = 0u; i < known_primes.size(); i += BLOCK_SIZE * BLOCK_SIZE)
		{
			std::cout << "Calculating " << i << std::endl;
			auto grid = dim3(BLOCK_COUNT);
			auto block = dim3(BLOCK_SIZE, BLOCK_SIZE);
			SieveParams params{ Array<bool>{cudaResult, result.size}, Array<size_t>{cudaKnownPrimes, known_primes.size()}, i};
			sieve_cuda <<< grid, block >>>(params);
			CUDA_STATUS_CHECK(cudaGetLastError());
			CUDA_STATUS_CHECK(cudaDeviceSynchronize());
		}
		std::cout << "Calculation done" << std::endl;

		std::cout << "Copying back result" << std::endl;
		CUDA_STATUS_CHECK(cudaMemcpy(result.ptr, cudaResult, result.size * sizeof(result[0]), cudaMemcpyDeviceToHost));
		std::cout << "Copying back done" << std::endl;

		CUDA_STATUS_CHECK(cudaFree(cudaKnownPrimes));
		CUDA_STATUS_CHECK(cudaFree(cudaResult));
		cudaDeviceReset();
	}
	else
	{
		constexpr auto BLOCK_COUNT = 1ll;
		constexpr auto BLOCK_SIZE = 8ll;

		for (auto i = 0u; i < known_primes.size(); i += BLOCK_SIZE)
		{
			std::cout << "Calculating " << i << std::endl;
			SieveParams params{ result, Array<size_t>::from_vector(known_primes), i };
			THREADABLE_CALL(BLOCK_SIZE, sieve_host, params);
		}
		std::cout << "Calculation done" << std::endl;
	}
}