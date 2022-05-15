#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

class Printer
{
	static constexpr auto PRIMES_PER_FILE = 1ll << 21;
	static constexpr auto COUT_SAMPLING_RATE = 1ll << 24;
	static constexpr auto FILE_PADDING = 4;

	size_t count = 0ull;
	bool coutPrint = false;
	std::stringstream cache;
	bool needsToWriteToFile = false;
	int fileIndex = 0;

	void writeToFile()
	{
		if (!needsToWriteToFile)
			return;

		std::stringstream ss;
		ss << std::setw(FILE_PADDING) << std::setfill('0') << std::to_string(fileIndex);
		std::ofstream file;
		file = std::ofstream("../output/prime_" + ss.str() + ".txt");
		file << cache.str();
		file.close();
		fileIndex++;
		cache = {};
		needsToWriteToFile = false;
	}

public:
	~Printer()
	{
		writeToFile();
	}

	template<bool print_to_file>
	void print(const Array<bool> result)
	{
		std::cout << "Printing results to file" << std::endl;
		for (auto i = 0ull; i < result.size; i++)
		{
			if (i % COUT_SAMPLING_RATE == 0) coutPrint = true;
			if (result[i])
			{
				count++;
				if (print_to_file)
				{
					cache << i << " ";
					if (count % PRIMES_PER_FILE == 0)
						writeToFile();
					else
						needsToWriteToFile = true;
				}
				if (coutPrint)
				{
					coutPrint = false;
					std::cout << i << " ";
				}
			}
		}
		std::cout << std::endl << "Printing done, count: " << count << std::endl;
	}
};
