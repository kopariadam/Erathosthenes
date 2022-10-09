#pragma once

constexpr auto ARRAY_SIZE = 1ull << 32;
static constexpr auto THREAD_COUNT = 8;
constexpr auto BLOCK_COUNT = 1u << 20;
constexpr auto BLOCK_SIZE = 16u;
static constexpr auto FILE_LENGHT = 1ll << 24;
static constexpr auto FILE_PADDING = 2;

constexpr size_t two_factor_sqrt(size_t n, size_t s = 0)
{
	constexpr auto LARGEST_INPUT = 1ull << 62;
	if (n > LARGEST_INPUT) return 1ull << 32;
	size_t i = 1ull << s;
	return i * i >= n ? i : two_factor_sqrt(n, s + 1);
}

constexpr int get_padding(int n, int p = 1, int s = 10)
{
	if (s > n) return p;
	return get_padding(n, p + 1, s * 10);
}