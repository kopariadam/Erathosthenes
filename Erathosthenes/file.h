#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>

class FileBase
{
public:
	void makeFolder()
	{
		_wmkdir(L"../output");
	}
	virtual std::ofstream makeFile(int index) = 0;
};

constexpr auto PASS_PADDING = get_padding(SIEVE_CALLS);
constexpr auto THREAD_PADDING = get_padding(THREAD_COUNT);
constexpr auto COMBINED_PADDING = FILE_PADDING + PASS_PADDING + THREAD_PADDING;

class SingleThreadedFile : public FileBase
{
public:
	std::ofstream makeFile(int index) override
	{
		std::stringstream ss;
		ss << std::setw(COMBINED_PADDING) << std::setfill('0') << std::to_string(index);
		return std::ofstream("../output/prime_" + ss.str() + ".txt", std::ios_base::out | std::ios_base::binary);
	}
};

class MultiThreadedFile : public FileBase
{
	int pass, thread;
public:
	MultiThreadedFile(int pass_, int thread_) : pass(pass_), thread(thread_) {}
	std::ofstream makeFile(int index) override
	{
		std::stringstream ss;
		ss << std::setfill('0');
		ss << std::setw(PASS_PADDING) << std::to_string(pass) << "_";
		ss << std::setw(THREAD_PADDING) << std::to_string(thread) << "_";
		ss << std::setw(FILE_PADDING) << std::to_string(index);
		return std::ofstream("../output/prime_" + ss.str() + ".txt", std::ios_base::out | std::ios_base::binary);
	}
};