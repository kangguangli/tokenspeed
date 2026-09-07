// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache/allocator/group_allocator.h"

// Build against either BlockPool version with the same compiler flags. Each
// process emits one raw sample; an external runner controls affinity, warmup,
// repeats, and alternating before/after order. No GPU work is involved.
int main(int argc, char** argv) {
    using namespace tokenspeed;
    using Clock = std::chrono::steady_clock;
    if (argc != 5) {
        std::cerr << "usage: block_pool_benchmark PARENTS PACKING BUCKETS FRAGMENTED_PARENTS\n";
        return 2;
    }
    const int parents = std::stoi(argv[1]);
    const int packing = std::stoi(argv[2]);
    const int buckets = std::stoi(argv[3]);
    const int fragmented = std::stoi(argv[4]);
    if (parents <= 0 || packing <= 0 || buckets <= 0 || packing % buckets != 0 || fragmented < 0 ||
        fragmented > parents || (fragmented > 0 && packing == 1)) {
        throw std::invalid_argument("invalid pool, packing, buckets, or fragmentation");
    }
    constexpr int requests = 16;
    constexpr int rounds = 64;
    BlockPool pool(parents);
    GroupAllocator allocator(packing, 0, buckets);
    std::vector<CacheBlockRef> seeds;
    if (fragmented > 0) {
        const auto seed_blocks = static_cast<std::int64_t>(fragmented) * packing;
        if (seed_blocks > INT32_MAX) {
            throw std::invalid_argument("seed allocation exceeds int32 block count");
        }
        const std::vector<std::int32_t> loads(static_cast<std::size_t>(buckets), 0);
        seeds = pool.AcquireBlocks(0, packing, static_cast<std::int32_t>(seed_blocks), loads);
        if (seeds.size() != static_cast<std::size_t>(seed_blocks)) {
            throw std::runtime_error("seed allocation failed");
        }
        // One hole per parent, spread across slots/owners. All bucket-index
        // configuration happens before the timed steady-state allocations.
        for (auto& block : seeds) {
            const auto location = block->Location();
            if (location.slot_index == (location.lcm_block_id - 1) % packing) {
                block.reset();
            }
        }
    }
    std::array<BlockTable, requests> tables;
    std::vector<std::int64_t> acquire_ns;
    std::vector<std::int64_t> release_ns;
    std::vector<CacheBlockLocation> locations;
    acquire_ns.reserve(requests * rounds);
    release_ns.reserve(requests);
    locations.reserve(requests * rounds);
    for (int round = 0; round < rounds; ++round) {
        for (auto& table : tables) {
            const auto start = Clock::now();
            const bool acquired = allocator.Acquire(pool, table, AcquirePlan{.num_blocks = 1});
            const auto finish = Clock::now();
            if (!acquired) {
                throw std::runtime_error("workload does not fit the pool");
            }
            acquire_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
            locations.push_back(table.Blocks().back()->Location());
        }
    }
    const auto used_parents = parents - pool.NumEmptyLcmBlocks();
    const auto occupied_slots = pool.NumOccupiedSlots();
    for (auto& table : tables) {
        const auto start = Clock::now();
        allocator.Free(table);
        const auto finish = Clock::now();
        release_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count());
    }
    const auto seed_cleanup_start = Clock::now();
    seeds.clear();
    const auto seed_cleanup_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - seed_cleanup_start).count();
    if (pool.NumOccupiedSlots() != 0 || pool.NumEmptyLcmBlocks() != parents) {
        throw std::runtime_error("cleanup left occupied blocks");
    }
    std::cout << "{\"parents\":" << parents << ",\"packing\":" << packing << ",\"buckets\":" << buckets
              << ",\"fragmented_parents\":" << fragmented << ",\"requests\":" << requests << ",\"rounds\":" << rounds
              << ",\"used_parents\":" << used_parents << ",\"occupied_slots\":" << occupied_slots
              << ",\"all_parents_released\":true,\"locations\":[";
    for (std::size_t i = 0; i < locations.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << '[' << locations[i].lcm_block_id << ',' << locations[i].slot_index << ']';
    }
    std::cout << "],\"acquire_ns\":[";
    for (std::size_t i = 0; i < acquire_ns.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << acquire_ns[i];
    }
    std::cout << "],\"free_request_ns\":[";
    for (std::size_t i = 0; i < release_ns.size(); ++i) {
        if (i > 0) {
            std::cout << ',';
        }
        std::cout << release_ns[i];
    }
    std::cout << "],\"seed_cleanup_ns\":" << seed_cleanup_ns << "}\n";
}
