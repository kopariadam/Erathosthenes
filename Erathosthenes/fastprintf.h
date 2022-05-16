#pragma once

static constexpr size_t TEN_POWS[] = {
	1ull,
	10ull,
	100ull,
	1000ull,
	10000ull,
	100000ull,
	1000000ull,
	10000000ull,
	100000000ull,
	1000000000ull,
	10000000000ull,
	100000000000ull,
	1000000000000ull,
	10000000000000ull,
	100000000000000ull,
	1000000000000000ull,
	10000000000000000ull,
	100000000000000000ull,
	1000000000000000000ull,
	10000000000000000000ull
};

#define countof(a) (sizeof(a) / sizeof(a[0]))

int get_index(size_t max)
{
	constexpr size_t n = countof(TEN_POWS);
	for (int i = 1; i < n; i++)
	{
		if (TEN_POWS[i] > max)
			return i - 1;
	}
	return n - 1;
}

char* fastsprintf(char* end, size_t number, int i)
{
	bool print = false;
	for (; i >= 0; i--)
	{
		if (TEN_POWS[i] <= number)
			print = true;
		if (print)
		{
			auto digit = number / TEN_POWS[i];
			number -= digit * TEN_POWS[i];
			*end = static_cast<char>(digit) + '0';
			end++;
		}
	}
	*end = ' ';
	end++;
	*end = '\0';
	return end;
}