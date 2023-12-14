#pragma once

constexpr auto GPU_ENABLED = true;
constexpr auto PRINT_TO_FILE = true;
constexpr auto SIEVE_CALLS = 16ull;

#define MULTI_THREADED_PRINTER 1
#define COUT_SAMPLING 0

constexpr auto MAIN_LOGS = true;
constexpr auto COMPUTE_LOGS = true;
constexpr auto PRINTER_LOGS = !MULTI_THREADED_PRINTER;

constexpr auto COUT_SAMPLING_RATE = 1ll << 24;
