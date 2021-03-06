#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include "fastprintf.h"

class FastStringStream
{
	std::vector<char> cache;
	char* end;

public:
	static constexpr auto OVER_RESERVE = 65ull;

	FastStringStream(size_t size)
	{
		cache.resize(size + OVER_RESERVE);
		reset();
	}

	bool canFit() { return end + OVER_RESERVE <= &cache.back(); }
	const char* c_str() { return cache.data(); }

	void write(size_t number, int index)
	{
		end = fastsprintf(end, number, index);
	}

	void reset()
	{
		cache[0] = '\0';
		end = cache.data();
	}

};

class Printer
{
	static constexpr auto FILE_LENGHT = 1ll << 24;
	static constexpr auto COUT_SAMPLING_RATE = 1ll << 24;
	static constexpr auto FILE_PADDING = 4;

	size_t count = 0ull;
	bool coutPrint = false;
	FastStringStream cache{ FILE_LENGHT };
	bool needsToWriteToFile = false;
	int fileIndex = 0;

public:
	Printer()
	{
		_wmkdir(L"../output");
	}
	void writeToFile()
	{
		if (!needsToWriteToFile)
			return;

		std::stringstream ss;
		ss << std::setw(FILE_PADDING) << std::setfill('0') << std::to_string(fileIndex);
		std::ofstream file = std::ofstream("../output/prime_" + ss.str() + ".txt", std::ios_base::out | std::ios_base::binary);
		file << cache.c_str();
		file.close();
		fileIndex++;
		cache.reset();
		needsToWriteToFile = false;
	}

	template<bool print_to_file>
	void print(const Array<bool> result, size_t offset)
	{
		auto startTime = std::chrono::high_resolution_clock::now();
		std::cout << "Printing results to file" << std::endl;

		if (offset == 0ull)
		{
			if (print_to_file)
				cache.write(2ull, get_index(2ull));
			count++;
		}

		auto index = get_index(offset + result.size);

		for (auto i = 1ull; i < result.size; i += 2)
		{
			size_t number = i + offset;
			if (number % COUT_SAMPLING_RATE == 1) coutPrint = true;
			if (result[i])
			{
				count++;
				if (print_to_file)
				{
					cache.write(number, index);
					if (!cache.canFit())
						writeToFile();
					else
						needsToWriteToFile = true;
				}
				if (coutPrint)
				{
					coutPrint = false;
					char buffer[FastStringStream::OVER_RESERVE];
					fastsprintf(buffer, number, index);
					std::cout << buffer;
				}
			}
		}
		std::cout << std::endl << "Printing done, count: " << count << std::endl;
		auto endTime = std::chrono::high_resolution_clock::now();
		auto calculationTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
		std::cout << "Printing took " << calculationTime << "s" << std::endl;
	}
};
