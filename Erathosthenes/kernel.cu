
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>

#define GPU_ENABLED 1
#define PRINT_KNOWN_PRIMES 0
#define PRINT_TO_FILE 0

cudaError_t cudaStatus;
#define CUDA_STATUS_CHECK(A) do {cudaStatus = (A); if(cudaStatus!=cudaSuccess) {std::cout<<"cuda error "<<__LINE__<<" "<<__FILE__<<"\n";}} while(0)

#include <thread>
#define THREADABLE_FUNCTION_START(name, ...)																				\
struct ThreadableFunction##name																								\
{																															\
	dim3 blockDim, threadIdx, blockIdx;																						\
	ThreadableFunction##name(unsigned int blockDim, unsigned int threadIdx)													\
		: blockDim(blockDim, 1, 1), threadIdx(threadIdx, 0, 0), blockIdx(0, 0, 0) {}										\
	void operator()( __VA_ARGS__ )																							\
	{


#define THREADABLE_FUNCTION_END																								\
	}																														\
};

#define THREADABLE_CALL(count, name, ...)																					\
do																															\
{																															\
	std::thread threads[count];																								\
	for(int qwertzuiop = 0; qwertzuiop < count; qwertzuiop++)																\
	{																														\
		threads[qwertzuiop] = std::thread(ThreadableFunction##name(count , qwertzuiop), __VA_ARGS__);						\
	}																														\
	for(int qwertzuiop = 0; qwertzuiop < count; qwertzuiop++)																\
	{																														\
		threads[qwertzuiop].join();																							\
	}																														\
} while (0)


constexpr auto SIZE_SHIFT = 32;
constexpr auto ARRAY_SIZE = 1ll << SIZE_SHIFT;
constexpr auto KNOWN_PRIME_LIMIT = 1ll << (SIZE_SHIFT / 2);

constexpr auto BLOCK_COUNT = 1ll << 20;
constexpr auto BLOCK_SIZE = 16ll;

constexpr auto CPU_BLOCK_COUNT = 1ll;
constexpr auto CPU_BLOCK_SIZE = 8ll;

constexpr auto PRIMES_PER_FILE = 1ll << 21;
constexpr auto COUT_SAMPLING_RATE = 1ll << 24;

__global__ void cudaSieve(bool* result, const unsigned int* known_primes, size_t starting_prime_idx, size_t prime_array_size)
{
	auto primeIdx = starting_prime_idx + blockDim.x * threadIdx.y + threadIdx.x;
	if (primeIdx >= prime_array_size) return;
	auto step = known_primes[primeIdx];

	auto numberIdx = blockIdx.x;
	auto numberBlockSize = ARRAY_SIZE / BLOCK_COUNT;
	auto startingNumber = numberBlockSize * numberIdx;
	if (startingNumber < KNOWN_PRIME_LIMIT) startingNumber = KNOWN_PRIME_LIMIT;
	startingNumber = ((startingNumber - 1) / step + 1) * step;
	auto endingNumber = numberBlockSize * (numberIdx + 1);

	for (auto i = startingNumber; i < endingNumber; i += step)
		result[i] = true;
}

//void hostSieve(bool* result, const unsigned int* known_primes, size_t starting_prime_idx, size_t prime_array_size)
//{
THREADABLE_FUNCTION_START(hostSieve, bool* result, const unsigned int* known_primes, size_t starting_prime_idx, size_t prime_array_size)
//auto primeIdx = starting_prime_idx;
auto primeIdx = starting_prime_idx + blockDim.x * threadIdx.y + threadIdx.x;
if (primeIdx >= prime_array_size) return;
auto step = known_primes[primeIdx];

//auto numberIdx = 0;
//auto numberBlockSize = ARRAY_SIZE;
auto numberIdx = blockIdx.x;
auto numberBlockSize = ARRAY_SIZE / CPU_BLOCK_COUNT;
auto startingNumber = numberBlockSize * numberIdx;
if (startingNumber < KNOWN_PRIME_LIMIT) startingNumber = KNOWN_PRIME_LIMIT;
startingNumber = ((startingNumber - 1) / step + 1) * step;
auto endingNumber = numberBlockSize * (numberIdx + 1);

for (auto i = startingNumber; i < endingNumber; i += step)
	result[i] = true;
THREADABLE_FUNCTION_END
//}

void print(const bool* result);

int main()
{
	auto startTime = std::chrono::high_resolution_clock::now();

	auto knownPrimes = std::vector<unsigned int>();

	std::cout << "Calculating base primes" << std::endl;
	knownPrimes.push_back(2u);
#if PRINT_KNOWN_PRIMES
	std::cout << 2 << " ";
#endif // PRINT_KNOWN_PRIMES
	for (auto i = 3u; i < KNOWN_PRIME_LIMIT; i += 2u)
	{
		auto isPrime = true;
		for (auto j = 3u; j * j <= i; j += 2u)
		{
			if (i % j == 0u)
			{
				isPrime = false;
				break;
			}
		}
		if (isPrime)
		{
			knownPrimes.push_back(i);
#if PRINT_KNOWN_PRIMES
			std::cout << i << " ";
#endif // PRINT_KNOWN_PRIMES
		}
	}
	std::cout << std::endl << "Base primes calculated, size: " << knownPrimes.size() << std::endl;

	auto result = (bool*)malloc(ARRAY_SIZE);
	memset(&result[0], 0, ARRAY_SIZE * sizeof(bool));
	memset(&result[0], 1, KNOWN_PRIME_LIMIT * sizeof(bool));
	for (auto p : knownPrimes)
		result[p] = false;

#if GPU_ENABLED
	std::cout << "Copying arrays to GPU" << std::endl;
	cudaSetDevice(0);

	unsigned int* cudaKnownPrimes;
	CUDA_STATUS_CHECK(cudaMalloc(&cudaKnownPrimes, knownPrimes.size() * sizeof(unsigned int)));
	CUDA_STATUS_CHECK(cudaMemcpy(cudaKnownPrimes, knownPrimes.data(), knownPrimes.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

	bool* cudaResult;
	CUDA_STATUS_CHECK(cudaMalloc(&cudaResult, ARRAY_SIZE));
	CUDA_STATUS_CHECK(cudaMemcpy(cudaResult, result, ARRAY_SIZE * sizeof(bool), cudaMemcpyHostToDevice));
	std::cout << "Arrays copied to GPU" << std::endl;

	for (auto i = 0u; i < knownPrimes.size(); i += BLOCK_SIZE * BLOCK_SIZE)
	{
		std::cout << "Calculating " << i << std::endl;
		auto grid = dim3(BLOCK_COUNT);
		auto block = dim3(BLOCK_SIZE, BLOCK_SIZE);
		cudaSieve << <grid, block >> > (cudaResult, cudaKnownPrimes, i, knownPrimes.size());
		CUDA_STATUS_CHECK(cudaGetLastError());
		CUDA_STATUS_CHECK(cudaDeviceSynchronize());
	}
	std::cout << "Calculation done" << std::endl;

	std::cout << "Copying back result" << std::endl;
	CUDA_STATUS_CHECK(cudaMemcpy(result, cudaResult, ARRAY_SIZE * sizeof(bool), cudaMemcpyDeviceToHost));
	std::cout << "Copying back done" << std::endl;

	CUDA_STATUS_CHECK(cudaFree(cudaKnownPrimes));
	CUDA_STATUS_CHECK(cudaFree(cudaResult));
	cudaDeviceReset();
#else
	for (auto i = 0u; i < knownPrimes.size(); i += CPU_BLOCK_SIZE)
	{
		std::cout << "Calculating " << i << std::endl;
		THREADABLE_CALL(CPU_BLOCK_SIZE, hostSieve, result, knownPrimes.data(), i, knownPrimes.size());
		//hostSieve(result, knownPrimes.data(), i, knownPrimes.size());
	}
	std::cout << "Calculation done" << std::endl;
#endif //GPU_ENABLED

	auto endTime = std::chrono::high_resolution_clock::now();
	auto calculationTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
	std::cout << "Took " << calculationTime << "ms" << std::endl;

	print(result);

	free(result);

	return 0;
}

void print(const bool* result)
{
	std::cout << "Printing results to file" << std::endl;
	std::ofstream file;
	auto count = 0;
	auto coutPrint = false;
	for (auto i = 0ull; i < ARRAY_SIZE; i++)
	{
		if (i % COUT_SAMPLING_RATE == 0) coutPrint = true;
		if (!result[i])
		{
#if PRINT_TO_FILE
			if (count % PRIMES_PER_FILE == 0)
			{
				if (file.is_open())
					file.close();
				std::stringstream ss;
				ss << std::setw(3) << std::setfill('0') << std::to_string(count / PRIMES_PER_FILE);
				file = std::ofstream("output/prime_" + ss.str() + ".txt");
			}
			file << i << " ";
#endif //PRINT_TO_FILE
			count++;
			if (coutPrint)
			{
				coutPrint = false;
				std::cout << i << " ";
			}
		}
	}
	if (file.is_open()) file.close();
	std::cout << std::endl << "Printing done, count: " << count << std::endl;
}