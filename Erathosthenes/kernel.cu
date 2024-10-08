﻿#include "hardware_defines.h"
#include "config.h"
#include "known_primes.h"
#include "sieve.cuh"
#include "multi_threaded_print.h"
#include "logger.h"

int main()
{
	auto startTime = std::chrono::high_resolution_clock::now();

	constexpr auto OFFSET_SIZE = ARRAY_SIZE * 2ull * ResultArray::value_per_byte();
	constexpr auto FINAL_NUMBER = OFFSET_SIZE * SIEVE_CALLS;
	main_log << "Calculating primes less than " << FINAL_NUMBER << " = 2^" << log2(FINAL_NUMBER) << "\n";

	auto knownPrimes = get_known_primes(two_factor_sqrt(OFFSET_SIZE));
	auto workingResult = ResultArray::from_malloc(malloc(ARRAY_SIZE), ARRAY_SIZE);
	auto  printingResult = ResultArray::from_malloc(malloc(ARRAY_SIZE), ARRAY_SIZE);
	Printer printer;
	for (auto offset = 0ull; offset < FINAL_NUMBER; offset += OFFSET_SIZE)
	{
		main_log << "\nSIEVE CALL " << (offset / OFFSET_SIZE + 1ull) << " OUT OF " << SIEVE_CALLS << "\n";
		sieve<GPU_ENABLED>(workingResult, offset, Array<uint32_t>::from_vector(knownPrimes));
		if (offset == 0ull)
			update_known_primes(knownPrimes, workingResult, two_factor_sqrt(FINAL_NUMBER));
		std::swap(workingResult, printingResult);
		printer.print<PRINT_TO_FILE>(printingResult, offset);
	}
	printer.writeToFile();
	free(workingResult.ptr);
	free(printingResult.ptr);

	auto endTime = std::chrono::high_resolution_clock::now();
	auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
	main_log << "Total run took " << totalTime << "s\n";

	return 0;
}
