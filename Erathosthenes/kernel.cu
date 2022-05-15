#include "known_primes.h"
#include "sieve.cuh"
#include "print.h"
#include <chrono>

#define GPU_ENABLED 1
#define PRINT_TO_FILE 1

int main()
{
	constexpr auto SIZE_SHIFT = 32;
	constexpr auto ARRAY_SIZE = 1ll << SIZE_SHIFT;
	constexpr auto KNOWN_PRIME_LIMIT = 1ll << (SIZE_SHIFT / 2);

	auto startTime = std::chrono::high_resolution_clock::now();

	auto knownPrimes = get_known_primes(KNOWN_PRIME_LIMIT);
	std::cout << std::endl << "Base primes calculated, size: " << knownPrimes.size() << std::endl;

	Array<bool> result;
	result.ptr = (bool*)malloc(ARRAY_SIZE);
	result.size = ARRAY_SIZE;
	sieve<GPU_ENABLED>(result, knownPrimes);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto calculationTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
	std::cout << "Took " << calculationTime << "ms" << std::endl;

	print<PRINT_TO_FILE>(result);

	free(result.ptr);
	return 0;
}
