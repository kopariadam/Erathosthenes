#include "known_primes.h"
#include "sieve.cuh"
#include "print.h"

#define GPU_ENABLED 1
#define PRINT_TO_FILE 1

constexpr size_t two_factor_sqrt(size_t n, size_t s = 0)
{
	size_t i = 1ull << s;
	return i * i >= n ? i : two_factor_sqrt(n, s + 1);
}

int main()
{
	constexpr auto ARRAY_SIZE = 1ull << 32;
	constexpr auto FINAL_NUMBER = 1ull << 32;

	auto knownPrimes = get_known_primes(two_factor_sqrt(ARRAY_SIZE));
	Array<bool> result;
	result.ptr = (bool*)malloc(ARRAY_SIZE);
	result.size = ARRAY_SIZE;
	Printer printer;
	for (auto offset = 0ull; offset < FINAL_NUMBER; offset += ARRAY_SIZE)
	{
		sieve<GPU_ENABLED>(result, /*offset,*/ knownPrimes/*, two_factor_sqrt(FINAL_NUMBER)*/);
		printer.print<PRINT_TO_FILE>(result/*, offset*/);
	}
	printer.writeToFile();
	free(result.ptr);

	return 0;
}
