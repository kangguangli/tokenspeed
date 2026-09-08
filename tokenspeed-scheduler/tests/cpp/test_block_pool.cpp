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

#include <gtest/gtest.h>

#include <array>
#include <deque>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>

#include "cache/core/block_pool.h"
#include "cache/allocator/group_allocator.h"

namespace tokenspeed::test {
namespace {

template <class T>
concept HasCacheIndex = requires(T& value) { value.ContainsCachedBlock("key"); };

static_assert(!HasCacheIndex<BlockPool>);

TEST(BlockPoolBucketTest, FreshAllocationsStayBalancedAtEveryLength) {
    for (std::int32_t buckets : {2, 4, 8}) {
        BlockPool pool(3);
        pool.RegisterGroup(0, 2 * buckets, buckets);
        GroupAllocator allocator(2 * buckets, 0, buckets);
        BlockTable request;
        std::vector<std::int32_t> counts(static_cast<std::size_t>(buckets), 0);
        for (std::int32_t i = 0; i < 6 * buckets; ++i) {
            ASSERT_TRUE(allocator.Acquire(pool, request, AcquirePlan{.num_blocks = 1}));
            ++counts[static_cast<std::size_t>(request.Blocks().back()->Location().slot_index % buckets)];
            const auto [low, high] = std::ranges::minmax_element(counts);
            EXPECT_LE(*high - *low, 1);
        }
        EXPECT_FALSE(allocator.Acquire(pool, request, AcquirePlan{.num_blocks = 1}));
        EXPECT_EQ(request.NumBlocks(), 6 * buckets);
    }
}

TEST(BlockPoolBucketTest, ChoosesBucketThenMostOccupiedParentAndLowestChild) {
    BlockPool pool(3);
    pool.RegisterGroup(0, 8, 2);
    auto existing = pool.AcquireBlocks(0, 8, 16);
    existing[1].reset();
    existing[9].reset();
    existing[10].reset();
    existing[11].reset();
    const std::vector<std::int32_t> loads{9, 0};
    auto next = pool.AcquireBlocks(0, 8, 3, loads);
    ASSERT_EQ(next.size(), 3u);
    EXPECT_EQ(next[0]->Location(), (CacheBlockLocation{1, 1}));
    EXPECT_EQ(next[1]->Location(), (CacheBlockLocation{2, 1}));
    EXPECT_EQ(next[2]->Location(), (CacheBlockLocation{2, 3}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
    EXPECT_EQ(loads, (std::vector<std::int32_t>{9, 0}));
}

TEST(BlockPoolBucketTest, ExhaustsAvailableHolesEvenWhenAnotherBucketIsLighter) {
    BlockPool pool(2);
    pool.RegisterGroup(0, 8, 2);
    auto existing = pool.AcquireBlocks(0, 8, 8);
    existing[4].reset();
    const std::vector<std::int32_t> loads{100, 0};
    auto next = pool.AcquireBlocks(0, 8, 1, loads);
    ASSERT_EQ(next.size(), 1u);
    EXPECT_EQ(next.front()->Location(), (CacheBlockLocation{1, 4}));
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(BlockPoolBucketTest, CapacityFailureLeavesAllLocationsAndLoadsUntouched) {
    BlockPool pool(1);
    pool.RegisterGroup(0, 8, 4);
    auto existing = pool.AcquireBlocks(0, 8, 5);
    const auto before = pool.OccupiedLocations(1);
    const std::vector<std::int32_t> loads{0, 2, 7, 0};
    auto failed = pool.AcquireBlocks(0, 8, 4, loads);
    EXPECT_TRUE(failed.empty());
    EXPECT_EQ(pool.OccupiedLocations(1), before);
    EXPECT_EQ(loads, (std::vector<std::int32_t>{0, 2, 7, 0}));
    auto success = pool.AcquireBlocks(0, 8, 3, loads);
    ASSERT_EQ(success.size(), 3u);
    EXPECT_EQ(pool.OccupiedCount(1), 8);
}

TEST(BlockPoolBucketTest, CountsSharedPrefixAndHeadroomButSkipsNullReferences) {
    BlockPool pool(1);
    pool.RegisterGroup(0, 8, 4);
    GroupAllocator allocator(8, 0, 4);
    BlockTable first;
    ASSERT_TRUE(allocator.Acquire(pool, first, AcquirePlan{.num_blocks = 3}));
    PrefixMatch hit;
    hit.blocks = {first.Blocks()[0], {}, first.Blocks()[2]};
    BlockTable second;
    allocator.ClaimHitBlocks(second, std::move(hit));
    ASSERT_TRUE(allocator.Acquire(pool, second, AcquirePlan{.num_blocks = 2}));
    EXPECT_EQ(second.Blocks()[3]->Location(), (CacheBlockLocation{1, 5}));
    EXPECT_EQ(second.Blocks()[4]->Location(), (CacheBlockLocation{1, 3}));
    EXPECT_EQ(first.Blocks()[0]->Location(), second.Blocks()[0]->Location());
    allocator.Free(first);
    EXPECT_FALSE(pool.AcquireBlock(1, 1));
    allocator.Free(second);
    auto rebound = pool.AcquireBlock(1, 1);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{1});
}

TEST(BlockPoolTest, ConstructsExactlyRequestedLcmBlocks) {
    BlockPool pool(8);
    EXPECT_EQ(pool.NumLcmBlocks(), 8);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 8);
}

TEST(BlockPoolTest, DestroyWithLiveReferenceReportsFatalInvariant) {
    EXPECT_DEATH(
        {
            spdlog::set_default_logger(spdlog::stderr_color_mt("fatal-check-test"));
            auto pool = std::make_unique<BlockPool>(1);
            CacheBlockRef ref = pool->AcquireBlock(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1);
            pool.reset();
        },
        "BlockPool destroyed with live block references");
}

TEST(BlockPoolTest, KOneBatchAcquireIsAllOrNothing) {
    BlockPool pool(3);
    auto blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 0);

    blocks = pool.AcquireBlocks(/*group_id=*/0, /*cache_blocks_per_lcm_block=*/1, /*num=*/3);
    EXPECT_EQ(blocks.size(), 3u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, EmptyParentBindsToGroupOnFirstChild) {
    BlockPool pool(3);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    EXPECT_EQ(first->Location(), (CacheBlockLocation{.lcm_block_id = 1, .slot_index = 0}));
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
}

TEST(BlockPoolLcmPlacementTest, RejectsPackingChangeWhileParentIsOccupied) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_THROW((void)pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/4), std::runtime_error);
}

TEST(BlockPoolLcmPlacementTest, NormalCapacityShortfallDoesNotMutatePartialParent) {
    BlockPool pool(1);
    CacheBlockRef existing = pool.AcquireBlock(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(existing);
    const CacheBlockLocation location = existing->Location();

    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/7, /*cache_blocks_per_lcm_block=*/2, /*num=*/2);

    EXPECT_TRUE(blocks.empty());
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{7});
    EXPECT_EQ(pool.OccupiedCount(1), 1);
    EXPECT_TRUE(pool.IsOccupied(location));
}

TEST(BlockPoolLcmPlacementTest, ParentRebindsOnlyAfterLastChildReleases) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/2);

    EXPECT_FALSE(pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8));
    child.reset();
    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);

    CacheBlockRef rebound = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/8);
    ASSERT_TRUE(rebound);
    EXPECT_EQ(rebound->Location().slot_index, 0);
    EXPECT_EQ(pool.BoundGroup(1), std::optional<std::uint32_t>{2});
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsRestoreBatchCapacity) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);
    ASSERT_EQ(blocks.size(), 2u);
    blocks[0].reset();
    blocks[1].reset();

    std::vector<CacheBlockRef> reused = pool.AcquireBlocks(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1, /*num=*/2);

    EXPECT_EQ(reused.size(), 2u);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, ReleasedParentsAreReusedInReleaseOrder) {
    BlockPool pool(4);
    std::vector<CacheBlockRef> blocks = pool.AcquireBlocks(/*group_id=*/1, /*cache_blocks_per_lcm_block=*/1, /*num=*/4);
    ASSERT_EQ(blocks.size(), 4u);
    const CacheBlockLocation first_released = blocks[0]->Location();
    const CacheBlockLocation second_released = blocks[2]->Location();
    blocks[0].reset();
    blocks[2].reset();

    CacheBlockRef first_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);
    CacheBlockRef second_reused = pool.AcquireBlock(/*group_id=*/2, /*cache_blocks_per_lcm_block=*/1);

    ASSERT_TRUE(first_reused);
    ASSERT_TRUE(second_reused);
    EXPECT_EQ(first_reused->Location(), first_released);
    EXPECT_EQ(second_reused->Location(), second_released);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 0);
}

TEST(BlockPoolLcmPlacementTest, IndependentChildrenShareOneParent) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef second = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);

    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first->Location().lcm_block_id, second->Location().lcm_block_id);
    EXPECT_NE(first->Location().slot_index, second->Location().slot_index);
    EXPECT_EQ(pool.OccupiedCount(1), 2);
}

TEST(BlockPoolLcmPlacementTest, ReleasingOneChildKeepsSiblingAndReusesOnlyItsSlot) {
    BlockPool pool(1);
    CacheBlockRef first = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    CacheBlockRef sibling = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    const CacheBlockLocation released = first->Location();
    const CacheBlockLocation retained = sibling->Location();

    first.reset();
    EXPECT_FALSE(pool.IsOccupied(released));
    EXPECT_TRUE(pool.IsOccupied(retained));

    CacheBlockRef replacement = pool.AcquireBlock(/*group_id=*/3, /*cache_blocks_per_lcm_block=*/2);
    ASSERT_TRUE(replacement);
    EXPECT_EQ(replacement->Location(), released);
    EXPECT_TRUE(pool.IsOccupied(retained));
}

TEST(BlockPoolLcmPlacementTest, FillsMostOccupiedPartialParentBeforeFreeParent) {
    BlockPool pool(3);
    auto first_parent = pool.AcquireBlocks(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3, /*num=*/3);
    ASSERT_EQ(first_parent.size(), 3u);
    CacheBlockRef second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(second_parent);
    first_parent.back().reset();

    CacheBlockRef fills_more_occupied_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_more_occupied_parent);
    EXPECT_EQ(fills_more_occupied_parent->Location().lcm_block_id, first_parent.front()->Location().lcm_block_id);

    CacheBlockRef fills_second_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent);
    CacheBlockRef fills_second_parent_again = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(fills_second_parent_again);
    CacheBlockRef uses_free_parent = pool.AcquireBlock(/*group_id=*/5, /*cache_blocks_per_lcm_block=*/3);
    ASSERT_TRUE(uses_free_parent);
    EXPECT_EQ(uses_free_parent->Location().lcm_block_id, 3);
}

TEST(BlockPoolLcmPlacementTest, LastReleaseImmediatelyClearsParentBinding) {
    BlockPool pool(1);
    CacheBlockRef child = pool.AcquireBlock(/*group_id=*/4, /*cache_blocks_per_lcm_block=*/8);
    child.reset();

    EXPECT_EQ(pool.BoundGroup(1), std::nullopt);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 1);
}

TEST(BlockPoolTest, AcquireUpToBlocksReturnsAvailableCompatiblePlacements) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/0, /*packing=*/2, /*max_num=*/3);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 4);
}

TEST(BlockPoolTest, AcquireUpToBlocksWithPackingOneReturnsPartialCapacity) {
    BlockPool pool(2);
    CacheBlockRef occupied = pool.AcquireBlock(/*group_id=*/0, /*packing=*/1);
    ASSERT_TRUE(occupied);

    std::vector<CacheBlockRef> partial = pool.AcquireUpToBlocks(/*group_id=*/1, /*packing=*/1, /*max_num=*/2);

    ASSERT_EQ(partial.size(), 1u);
    EXPECT_EQ(pool.BoundGroup(partial.front()->Location().lcm_block_id), 1u);
    EXPECT_EQ(pool.NumOccupiedSlots(), 2);
}

TEST(BlockPoolTest, ExactAcquireBlocksRemainsAllOrNothing) {
    BlockPool pool(2);
    std::vector<CacheBlockRef> first = pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/3);
    ASSERT_EQ(first.size(), 3u);

    EXPECT_TRUE(pool.AcquireBlocks(/*group_id=*/0, /*packing=*/2, /*num=*/2).empty());
    EXPECT_EQ(pool.NumOccupiedSlots(), 3);
}

TEST(BlockPoolBucketTest, RetractionWithSharedPrefixRestoresHolesAndParentFifo) {
    BlockPool pool(3);
    pool.RegisterGroup(0, 8, 4);
    GroupAllocator allocator(8, 0, 4);
    BlockTable request;
    ASSERT_TRUE(allocator.Acquire(pool, request, AcquirePlan{.num_blocks = 10}));
    PrefixMatch prefix;
    prefix.blocks.assign(request.Blocks().begin(), request.Blocks().begin() + 4);
    allocator.Free(request);
    EXPECT_EQ(pool.OccupiedCount(1), 4);
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

    allocator.ClaimHitBlocks(request, std::move(prefix));
    ASSERT_TRUE(allocator.Acquire(pool, request, AcquirePlan{.num_blocks = 4}));
    for (std::int32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(request.Blocks()[static_cast<std::size_t>(i)]->Location(), (CacheBlockLocation{1, i}));
    }
    allocator.Free(request);
    auto rebound = pool.AcquireBlocks(1, 1, 3);
    ASSERT_EQ(rebound.size(), 3u);
    EXPECT_EQ(rebound[0]->Location().lcm_block_id, 3);
    EXPECT_EQ(rebound[1]->Location().lcm_block_id, 2);
    EXPECT_EQ(rebound[2]->Location().lcm_block_id, 1);
}

TEST(BlockPoolBucketTest, FixedShardsCoverFullParentsAndLaterRebinding) {
    BlockPool pool(2);
    pool.RegisterGroup(7, 8, 4);
    auto full = pool.AcquireBlocks(7, 8, 8);
    const std::array<std::int32_t, 4> loads{0, 0, 0, 0};
    auto other = pool.AcquireBlocks(7, 8, 1, loads);
    full[6].reset();
    auto hole = pool.AcquireBlocks(7, 8, 1, std::array<std::int32_t, 4>{3, 3, 0, 3});
    ASSERT_EQ(hole.size(), 1u);
    EXPECT_EQ(hole.front()->Location(), (CacheBlockLocation{1, 6}));
    full.clear();
    other.clear();
    hole.clear();
    EXPECT_EQ(pool.NumEmptyLcmBlocks(), 2);

    pool.RegisterGroup(7, 8, 4);
    EXPECT_THROW(pool.RegisterGroup(7, 8, 2), std::runtime_error);
    EXPECT_THROW(pool.RegisterGroup(7, 2, 4), std::runtime_error);
    auto rebound = pool.AcquireBlocks(7, 8, 16, loads);
    EXPECT_EQ(rebound.size(), 16u);
    EXPECT_TRUE(pool.AcquireBlocks(7, 8, 1, loads).empty());
}

// A deliberately exhaustive oracle: rank every compatible free slot from
// scratch. It shares neither the availability indices nor the capacity
// counters of BlockPool, and tracks physical lifetime via reference counts.
class PlacementOracle {
public:
    explicit PlacementOracle(std::int32_t parents) : parents_(static_cast<std::size_t>(parents)) {
        for (std::int32_t id = 1; id <= parents; ++id) {
            empty_.push_back(id);
        }
    }

    std::vector<CacheBlockLocation> Acquire(std::uint32_t group, std::int32_t packing, std::int32_t count,
                                            std::vector<std::int32_t> loads = {}, bool partial = false) {
        auto planned = parents_;
        auto empty = empty_;
        std::vector<CacheBlockLocation> result;
        for (std::int32_t i = 0; i < count; ++i) {
            using Priority = std::tuple<std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t>;
            std::optional<Priority> best;
            CacheBlockLocation chosen;
            for (std::size_t p = 0; p < planned.size(); ++p) {
                const Parent& parent = planned[p];
                if (parent.group != group) {
                    continue;
                }
                const auto occupied =
                    static_cast<std::int32_t>(std::ranges::count_if(parent.refs, [](int refs) { return refs > 0; }));
                for (std::int32_t slot = 0; slot < packing; ++slot) {
                    if (parent.refs[static_cast<std::size_t>(slot)] > 0) {
                        continue;
                    }
                    const auto bucket = loads.empty() ? 0 : slot % static_cast<std::int32_t>(loads.size());
                    const auto id = static_cast<std::int32_t>(p + 1);
                    const Priority priority{loads.empty() ? 0 : loads[static_cast<std::size_t>(bucket)], bucket,
                                            -occupied, id, slot};
                    if (!best || priority < *best) {
                        best = priority;
                        chosen = CacheBlockLocation{id, slot};
                    }
                }
            }
            if (!best) {
                if (empty.empty()) {
                    if (!partial) {
                        return {};
                    }
                    break;
                }
                const auto slot =
                    loads.empty() ? 0 : static_cast<std::int32_t>(std::ranges::min_element(loads) - loads.begin());
                chosen = CacheBlockLocation{empty.front(), slot};
                empty.pop_front();
                Parent& parent = planned[static_cast<std::size_t>(chosen.lcm_block_id - 1)];
                parent.group = group;
                parent.refs.assign(static_cast<std::size_t>(packing), 0);
            }
            planned[static_cast<std::size_t>(chosen.lcm_block_id - 1)]
                .refs[static_cast<std::size_t>(chosen.slot_index)] = 1;
            if (!loads.empty()) {
                ++loads[static_cast<std::size_t>(chosen.slot_index) % loads.size()];
            }
            result.push_back(chosen);
        }
        parents_ = std::move(planned);
        empty_ = std::move(empty);
        return result;
    }

    void Retain(CacheBlockLocation location) {
        ++parents_[static_cast<std::size_t>(location.lcm_block_id - 1)]
              .refs[static_cast<std::size_t>(location.slot_index)];
    }

    void Release(CacheBlockLocation location) {
        Parent& parent = parents_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        --parent.refs[static_cast<std::size_t>(location.slot_index)];
        if (std::ranges::all_of(parent.refs, [](int refs) { return refs == 0; })) {
            parent.group.reset();
            parent.refs.clear();
            empty_.push_back(location.lcm_block_id);
        }
    }

    void Check(const BlockPool& pool) const {
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), static_cast<std::int32_t>(empty_.size()));
        for (std::size_t p = 0; p < parents_.size(); ++p) {
            const Parent& parent = parents_[p];
            const auto id = static_cast<std::int32_t>(p + 1);
            EXPECT_EQ(pool.BoundGroup(id), parent.group);
            EXPECT_EQ(pool.OccupiedCount(id), std::ranges::count_if(parent.refs, [](int refs) { return refs > 0; }));
            for (std::size_t slot = 0; slot < parent.refs.size(); ++slot) {
                EXPECT_EQ(pool.IsOccupied(CacheBlockLocation{id, static_cast<std::int32_t>(slot)}),
                          parent.refs[slot] > 0);
            }
        }
    }

private:
    struct Parent {
        std::optional<std::uint32_t> group;
        std::vector<int> refs;
    };
    std::vector<Parent> parents_;
    std::deque<std::int32_t> empty_;
};

TEST(BlockPoolBucketTest, InterleavedLifetimesMatchExhaustivePlacementOracle) {
    for (std::uint32_t seed : {0u, 17u, 2026u}) {
        SCOPED_TRACE(seed);
        std::mt19937 random(seed);
        BlockPool pool(24);
        PlacementOracle oracle(24);
        std::vector<CacheBlockRef> held;
        const std::array<std::int32_t, 3> packing{8, 16, 1};
        const std::array<std::int32_t, 3> shards{2, 8, 1};
        for (std::uint32_t group = 0; group < packing.size(); ++group) {
            pool.RegisterGroup(group, packing[group], shards[group]);
        }
        for (int step = 0; step < 1200; ++step) {
            SCOPED_TRACE(step);
            const auto operation = random() % 10;
            if (operation < 5 || held.empty()) {
                const auto group = random() % 3;
                const auto count = static_cast<std::int32_t>(1 + random() % 20);
                const bool partial = random() % 4 == 0;
                std::vector<std::int32_t> loads;
                if (!partial && group != 2 && random() % 4 != 0) {
                    loads.resize(static_cast<std::size_t>(shards[group]));
                    for (auto& load : loads) {
                        load = static_cast<std::int32_t>(random() % 31);
                    }
                }
                const auto saved_loads = loads;
                const auto expected = oracle.Acquire(group, packing[group], count, loads, partial);
                auto actual = partial ? pool.AcquireUpToBlocks(group, packing[group], count)
                                      : pool.AcquireBlocks(group, packing[group], count, loads);
                ASSERT_EQ(actual.size(), expected.size());
                EXPECT_EQ(loads, saved_loads);
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    EXPECT_EQ(actual[i]->Location(), expected[i]);
                    held.push_back(std::move(actual[i]));
                }
            } else if (operation == 9) {
                const auto index = random() % held.size();
                oracle.Retain(held[index]->Location());
                CacheBlockRef copy = held[index];
                held.push_back(std::move(copy));
            } else {
                const auto index = random() % held.size();
                oracle.Release(held[index]->Location());
                held[index].reset();
                held[index] = std::move(held.back());
                held.pop_back();
            }
            oracle.Check(pool);
        }
        while (!held.empty()) {
            oracle.Release(held.back()->Location());
            held.pop_back();
        }
        oracle.Check(pool);
        EXPECT_EQ(pool.NumEmptyLcmBlocks(), pool.NumLcmBlocks());
    }
}

}  // namespace
}  // namespace tokenspeed::test
