#pragma once
#include <iostream>

struct Logger
{
	bool enabled;
	template<typename T>
	Logger& operator<<(const T &data)
	{
		if (enabled)
			std::cout << data;
		return *this;
	}
};

extern Logger main_log;
extern Logger compute_log;
extern Logger printer_log;