#include "known_primes.h"
#include "sieve.cuh"
#include "multi_threaded_print.h"

#define GPU_ENABLED 1
#define PRINT_TO_FILE 1

constexpr size_t two_factor_sqrt(size_t n, size_t s = 0)
{
	constexpr auto LARGEST_INPUT = 1ull << 62;
	if (n > LARGEST_INPUT) return 1ull << 32;
	size_t i = 1ull << s;
	return i * i >= n ? i : two_factor_sqrt(n, s + 1);
}

int main()
{
	constexpr auto ARRAY_SIZE = 1ull << 32;
	constexpr auto FINAL_NUMBER = 1ull << 36;
	constexpr auto SIEVE_CALLS = FINAL_NUMBER / ARRAY_SIZE;

	auto knownPrimes = get_known_primes(two_factor_sqrt(ARRAY_SIZE));
	Array<bool> workingResult{ (bool*)malloc(ARRAY_SIZE), ARRAY_SIZE };
	Array<bool> printingResult{ (bool*)malloc(ARRAY_SIZE), ARRAY_SIZE };
	MultiThreadedPrinter printer;
	for (auto offset = 0ull; offset < FINAL_NUMBER; offset += ARRAY_SIZE)
	{
		std::cout << "\nSIEVE CALL " << (offset / ARRAY_SIZE + 1ull) << " OUT OF " << SIEVE_CALLS << std::endl;
		sieve<GPU_ENABLED>(workingResult, offset, knownPrimes);
		if (offset == 0ull)
			update_known_primes(knownPrimes, workingResult, two_factor_sqrt(FINAL_NUMBER));
		std::swap(workingResult, printingResult);
		printer.print<PRINT_TO_FILE>(printingResult, offset);
	}
	printer.writeToFile();
	free(workingResult.ptr);
	free(printingResult.ptr);

	return 0;
}
