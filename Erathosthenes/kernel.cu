#include "hardware_defines.h"
#include "config.h"
#include "known_primes.h"
#include "sieve.cuh"
#include "multi_threaded_print.h"
#include "logger.h"

int main()
{
	constexpr auto OFFSET_SIZE = ARRAY_SIZE * 2ull;
	constexpr auto FINAL_NUMBER = OFFSET_SIZE * SIEVE_CALLS;

	auto knownPrimes = get_known_primes(two_factor_sqrt(OFFSET_SIZE));
	Array<bool> workingResult{ (bool*)malloc(ARRAY_SIZE), ARRAY_SIZE };
	Array<bool> printingResult{ (bool*)malloc(ARRAY_SIZE), ARRAY_SIZE };
	Printer printer;
	for (auto offset = 0ull; offset < FINAL_NUMBER; offset += OFFSET_SIZE)
	{
		main_log << "\nSIEVE CALL " << (offset / OFFSET_SIZE + 1ull) << " OUT OF " << SIEVE_CALLS << "\n";
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
