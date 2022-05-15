#pragma once
#include <vector>
#include <iostream>
#include "parameters.cuh"

std::vector<size_t> get_known_primes(size_t limit)
{
	std::cout << "Calculating base primes" << std::endl;

	auto knownPrimes = std::vector<size_t>();
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
	std::cout << std::endl << "Base primes calculated, size: " << knownPrimes.size() << std::endl;
	return knownPrimes;
}

void update_known_primes(std::vector<size_t>& known_primes, const Array<bool>& result, size_t limit)
{
	for (size_t i = known_primes.back(); i < limit; i++)
	{
		if (result[i])
			known_primes.push_back(i);
	}
}