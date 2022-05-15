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

	FastStringStream& operator<<(const char *s)
	{
		strcpy(end, s);
		end += strlen(s);
		return *this;
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
	void print(const Array<bool> result)
	{
		auto startTime = std::chrono::high_resolution_clock::now();

		std::cout << "Printing results to file" << std::endl;
		for (auto i = 0ull; i < result.size; i++)
		{
			if (i % COUT_SAMPLING_RATE == 0) coutPrint = true;
			if (result[i])
			{
				count++;
				if (print_to_file)
				{
					char buffer[FastStringStream::OVER_RESERVE];
					std::sprintf(buffer, "%llu ", i);
					cache << buffer;
					if (!cache.canFit())
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
		auto endTime = std::chrono::high_resolution_clock::now();
		auto calculationTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
		std::cout << "Printing took " << calculationTime << "s" << std::endl;
	}
};
