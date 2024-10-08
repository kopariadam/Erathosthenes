#pragma once
#include "fastprintf.h"
#include <memory>
#include "file.h"
#include "parameters.cuh"
#include "logger.h"
#include "hardware_defines.h"
#include "config.h"

class FastStringStream
{
	std::vector<char> cache;
	char* end;

public:
	static constexpr auto OVER_RESERVE = 21ull;

	FastStringStream(size_t size)
	{
		cache.resize(size);
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

class SingleThreadedPrinter
{
	size_t count = 0ull;
	#if COUT_SAMPLING
	bool coutPrint = false;
	#endif
	FastStringStream cache{ FILE_LENGHT };
	bool needsToWriteToFile = false;
	int fileIndex = 0;

	std::unique_ptr<FileBase> fileMaker;

public:
	SingleThreadedPrinter(std::unique_ptr<FileBase>&& file_maker = std::make_unique<SingleThreadedFile>()) : fileMaker(std::move(file_maker))
	{
		fileMaker->makeFolder();
	}
	size_t getCount() { return count; }
	void writeToFile()
	{
		if (!needsToWriteToFile)
			return;

		auto file = fileMaker->makeFile(fileIndex);
		file << cache.c_str();
		file.close();
		fileIndex++;
		cache.reset();
		needsToWriteToFile = false;
	}

	template<bool print_to_file>
	void print(const ResultArray result, size_t offset)
	{
		auto startTime = std::chrono::high_resolution_clock::now();
		printer_log << "Printing results to file\n";

		if (offset == 0ull)
		{
			if (print_to_file)
				cache.write(2ull, get_index(2ull));
			count++;
		}

		auto index = get_index(offset + result.size * 2ull);

		for (auto i = 0ull; i < result.size; i++)
		{
			size_t number = i * 2ull + 1ull + offset;
			#if COUT_SAMPLING
			if (number % COUT_SAMPLING_RATE == 1) coutPrint = true;
			#endif
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
				#if COUT_SAMPLING
				if (coutPrint)
				{
					coutPrint = false;
					char buffer[FastStringStream::OVER_RESERVE];
					fastsprintf(buffer, number, index);
					printer_log << buffer;
				}
				#endif
			}
		}
		printer_log << "\nPrinting done, count: " << count << "\n";
		auto endTime = std::chrono::high_resolution_clock::now();
		auto calculationTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
		printer_log << "Printing took " << calculationTime << "s\n";
	}
};
