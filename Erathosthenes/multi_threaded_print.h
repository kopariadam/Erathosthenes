#pragma once
#include "hardware_defines.h"
#include "print.h"
#include <thread>
#include <array>
#include <atomic>

template<bool print_to_file>
void print_part(int pass, int thread, const ResultArray result, size_t offset, std::atomic_uint64_t *count)
{
	SingleThreadedPrinter printer{ std::make_unique<MultiThreadedFile>(pass, thread) };
	printer.print<print_to_file>(result, offset);
	printer.writeToFile();
	*count += printer.getCount();
}

class MultiThreadedPrinter
{
	int pass = 0;
	std::array<std::thread, THREAD_COUNT> threads;
	bool threadsStarted = false;
	std::atomic_uint64_t count{ 0 };

public:
	template<bool print_to_file>
	void print(const ResultArray result, size_t offset)
	{
		writeToFile();
		size_t size = result.size / THREAD_COUNT;
		for (int i = 0; i < THREAD_COUNT; i++)
		{
			const ResultArray threadResult = result.subarray(i * size, size);
			size_t threadOffset = offset + i * size * 2ull;
			threads[i] = std::thread(print_part<print_to_file>, pass, i, threadResult, threadOffset, &count);
		}
		pass++;
		threadsStarted = true;
	}
	void writeToFile()
	{
		if (threadsStarted)
		{
			for (auto& thread : threads)
				thread.join();
			main_log << "\nPrinting for pass " << pass << " done, count: " << count << "\n";
			threadsStarted = false;
		}
	}
};

#if MULTI_THREADED_PRINTER
using Printer = MultiThreadedPrinter;
#else
using Printer = SingleThreadedPrinter;
#endif