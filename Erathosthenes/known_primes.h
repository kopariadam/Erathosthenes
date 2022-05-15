#pragma once
#include <vector>
#include <iostream>

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
	return knownPrimes;
}