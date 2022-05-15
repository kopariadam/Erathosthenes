#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

template<bool print_to_file>
void print(const Array<bool> result)
{
	constexpr auto PRIMES_PER_FILE = 1ll << 21;
	constexpr auto COUT_SAMPLING_RATE = 1ll << 24;

	std::cout << "Printing results to file" << std::endl;
	std::ofstream file;
	auto count = 0ull;
	auto coutPrint = false;
	for (auto i = 0ull; i < result.size; i++)
	{
		if (i % COUT_SAMPLING_RATE == 0) coutPrint = true;
		if (result[i])
		{
			if (print_to_file)
			{
				if (count % PRIMES_PER_FILE == 0)
				{
					if (file.is_open())
						file.close();
					std::stringstream ss;
					ss << std::setw(4) << std::setfill('0') << std::to_string(count / PRIMES_PER_FILE);
					file = std::ofstream("output/prime_" + ss.str() + ".txt");
				}
				file << i << " ";
			}
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