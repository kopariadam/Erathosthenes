#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

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

	bool canFit()
	{
		return end + OVER_RESERVE <= &cache.back();
	}

	void write(const char *s, int len)
	{
		strcpy(end, s);
		end += len;
	}

	void reset()
	{
		cache[0] = '\0';
		end = cache.data();
	}

	const char* c_str()
	{
		return cache.data();
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
	void writeToFile()
	{
		if (!needsToWriteToFile)
			return;

		std::stringstream ss;
		ss << std::setw(FILE_PADDING) << std::setfill('0') << std::to_string(fileIndex);
		std::ofstream file;
		file = std::ofstream("../output/prime_" + ss.str() + ".txt");
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
			{
				auto two = "2 ";
				cache.write(two, strlen(two));
			}
			count++;
		}

		for (auto i = 1ull; i < result.size; i += 2)
		{
			size_t number = i + offset;
			if (number % COUT_SAMPLING_RATE == 1) coutPrint = true;
			if (result[i])
			{
				count++;
				if (print_to_file)
				{
					char buffer[FastStringStream::OVER_RESERVE];
					int length = std::sprintf(buffer, "%llu ", number);
					cache.write(buffer, length);
					if (!cache.canFit())
						writeToFile();
					else
						needsToWriteToFile = true;
				}
				if (coutPrint)
				{
					coutPrint = false;
					std::cout << number << " ";
				}
			}
		}
		std::cout << std::endl << "Printing done, count: " << count << std::endl;
		auto endTime = std::chrono::high_resolution_clock::now();
		auto calculationTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
		std::cout << "Printing took " << calculationTime << "s" << std::endl;
	}
};
