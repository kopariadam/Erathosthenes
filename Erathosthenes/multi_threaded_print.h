#pragma once
#include "hardware_defines.h"
#include "print.h"
#include <thread>
#include <array>

template<bool print_to_file>
void print_part(int pass, int thread, const Array<bool> result, size_t offset)
{
	SingleThreadedPrinter printer{ std::make_unique<MultiThreadedFile>(pass, thread) };
	printer.print<print_to_file>(result, offset);
	printer.writeToFile();
}

class MultiThreadedPrinter
{
	int pass = 0;
	std::array<std::thread, THREAD_COUNT> threads;
	bool threadsStarted = false;

public:
	template<bool print_to_file>
	void print(const Array<bool> result, size_t offset)
	{
		writeToFile();
		size_t size = result.size / THREAD_COUNT;
		for (int i = 0; i < THREAD_COUNT; i++)
		{
			Array<bool> threadResult;
			threadResult.ptr = result.ptr + i * size;
			threadResult.size = size;
			size_t threadOffset = offset + i * size * 2ull;
			threads[i] = std::thread(print_part<print_to_file>, pass, i, threadResult, threadOffset);
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
			threadsStarted = false;
		}
	}
};

#if MULTI_THREADED_PRINTER
using Printer = MultiThreadedPrinter;
#else
using Printer = SingleThreadedPrinter;
#endif