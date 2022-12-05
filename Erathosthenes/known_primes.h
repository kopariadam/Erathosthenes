#pragma once
#include <vector>
#include <iostream>
#include "parameters.cuh"
#include <math.h>
#include "logger.h"

std::vector<uint32_t> get_known_primes(size_t limit)
{
	compute_log << "Calculating base primes\n";

	auto knownPrimes = std::vector<uint32_t>();
	knownPrimes.reserve(limit / static_cast<size_t>(log(limit)));
	knownPrimes.push_back(2u);
	for (auto i = 3u; i < limit; i += 2u)
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
		}
	}
	compute_log << "\nBase primes calculated, size: " << knownPrimes.size() << "\n";
	return knownPrimes;
}

void update_known_primes(std::vector<uint32_t>& known_primes, const Array<bool>& result, size_t limit)
{
	known_primes.reserve(limit / static_cast<size_t>(log(limit)));
	for (auto i = known_primes.back() + 2u; i < limit; i += 2u)
	{
		if (result[i / 2u])
			known_primes.push_back(i);
	}
}