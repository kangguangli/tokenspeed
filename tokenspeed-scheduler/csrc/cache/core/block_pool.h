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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "cache/core/cache_block_ref.h"
#include "utils.h"

namespace tokenspeed {

// Physical LCM placement only. It deliberately has no cache key, LRU node,
// CacheBlock pointer, or ownership count.
class BlockPool {
public:
    explicit BlockPool(std::int32_t num_lcm_blocks) : lcm_blocks_(checkedLcmBlockCount(num_lcm_blocks)) {
        for (std::int32_t id = 1; id <= num_lcm_blocks; ++id) {
            free_parent_ids_.push_back(id);
        }
    }

    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    ~BlockPool() noexcept { FatalCheck(NumOccupiedSlots() == 0, "BlockPool destroyed with live block references"); }

    // Number of physical LCM blocks. Kernel page 0 is reserved separately.
    std::int32_t NumLcmBlocks() const noexcept { return static_cast<std::int32_t>(lcm_blocks_.size()); }
    std::int32_t NumEmptyLcmBlocks() const noexcept { return static_cast<std::int32_t>(free_parent_ids_.size()); }

    CacheBlockRef AcquireBlock(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block) {
        std::vector<CacheBlockRef> blocks = AcquireBlocks(group_id, cache_blocks_per_lcm_block, 1);
        if (blocks.empty()) {
            return {};
        }
        return std::move(blocks.front());
    }

    std::vector<CacheBlockRef> AcquireBlocks(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block,
                                             std::int32_t num, std::span<const std::int32_t> bucket_loads = {}) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (num <= 0) {
            return {};
        }

        if (bucket_loads.size() > 1) {
            _assert(static_cast<std::size_t>(cache_blocks_per_lcm_block) % bucket_loads.size() == 0,
                    "allocation buckets must divide parent packing");
            for (std::int32_t load : bucket_loads) {
                _assert(load >= 0, "allocation bucket loads must be non-negative");
            }
        }
        if (availableBlocks(group_id, cache_blocks_per_lcm_block) < static_cast<std::size_t>(num)) {
            return {};
        }
        return acquireAvailableBlocks(group_id, cache_blocks_per_lcm_block, num, bucket_loads);
    }

    std::vector<CacheBlockRef> AcquireUpToBlocks(std::uint32_t group_id, std::int32_t cache_blocks_per_lcm_block,
                                                 std::int32_t max_num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (max_num <= 0) {
            return {};
        }
        const auto take = static_cast<std::int32_t>(
            std::min(static_cast<std::size_t>(max_num), availableBlocks(group_id, cache_blocks_per_lcm_block)));
        return take > 0 ? acquireAvailableBlocks(group_id, cache_blocks_per_lcm_block, take)
                        : std::vector<CacheBlockRef>{};
    }

    std::vector<CacheBlockRef> AcquireAvailableBlocksInOrder(std::span<const std::uint32_t> group_ids,
                                                             std::span<const std::int32_t> cache_blocks_per_group) {
        for (std::uint32_t group_id : group_ids) {
            _assert(group_id < cache_blocks_per_group.size(), "group id has no packing");
            _assert(cache_blocks_per_group[group_id] > 0, "cache_blocks_per_lcm_block must be > 0");
            (void)availableBlocks(group_id, cache_blocks_per_group[group_id]);
        }
        std::vector<CacheBlockRef> out(group_ids.size());
        for (std::size_t i = 0; i < group_ids.size(); ++i) {
            out[i] = AcquireBlock(group_ids[i], cache_blocks_per_group[group_ids[i]]);
        }
        return out;
    }

    std::vector<CacheBlockRef> AcquireUpToBlocksFromEmptyParent(std::uint32_t group_id,
                                                                std::int32_t cache_blocks_per_lcm_block,
                                                                std::int32_t lcm_block_id, std::int32_t max_num) {
        _assert(cache_blocks_per_lcm_block > 0, "cache_blocks_per_lcm_block must be > 0");
        if (max_num <= 0) {
            return {};
        }
        const LcmBlock& parent = lcmBlock(lcm_block_id);
        _assert(parent.occupied_count == 0 && !parent.bound_group, "directed Host parent must be empty");
        _assert(!free_parent_ids_.empty() && free_parent_ids_.front() == lcm_block_id,
                "directed Host parent must be the next free parent");

        const std::int32_t take = std::min(max_num, cache_blocks_per_lcm_block);
        (void)prepareAvailability(group_id, cache_blocks_per_lcm_block);
        std::vector<CacheBlockRef> out;
        out.reserve(static_cast<std::size_t>(take));
        for (std::int32_t slot = 0; slot < take; ++slot) {
            out.push_back(createBlockRef(group_id, cache_blocks_per_lcm_block,
                                         CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = slot}));
        }
        return out;
    }

    std::optional<std::uint32_t> BoundGroup(std::int32_t lcm_block_id) const {
        return lcmBlock(lcm_block_id).bound_group;
    }
    std::int32_t OccupiedCount(std::int32_t lcm_block_id) const {
        return static_cast<std::int32_t>(lcmBlock(lcm_block_id).occupied_count);
    }
    bool IsOccupied(CacheBlockLocation location) const {
        const LcmBlock& lcm_block = lcmBlock(location.lcm_block_id);
        return location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < lcm_block.occupancy.size() &&
               lcm_block.occupancy[static_cast<std::size_t>(location.slot_index)];
    }
    std::int32_t NumOccupiedSlots() const noexcept {
        std::int32_t count = 0;
        for (const LcmBlock& block : lcm_blocks_) {
            count += static_cast<std::int32_t>(block.occupied_count);
        }
        return count;
    }
    std::vector<CacheBlockLocation> OccupiedLocations(std::int32_t lcm_block_id) const {
        const LcmBlock& lcm_block = lcmBlock(lcm_block_id);
        std::vector<CacheBlockLocation> locations;
        locations.reserve(lcm_block.occupied_count);
        for (std::size_t slot = 0; slot < lcm_block.occupancy.size(); ++slot) {
            if (lcm_block.occupancy[slot]) {
                locations.push_back(
                    CacheBlockLocation{.lcm_block_id = lcm_block_id, .slot_index = static_cast<std::int32_t>(slot)});
            }
        }
        return locations;
    }

    void Release(CacheBlockLocation location) noexcept {
        FatalCheck(location.lcm_block_id > 0 && static_cast<std::size_t>(location.lcm_block_id) <= lcm_blocks_.size(),
                   "CacheBlock location has invalid LCM block id");
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        FatalCheck(location.slot_index >= 0 && static_cast<std::size_t>(location.slot_index) < parent.occupancy.size(),
                   "CacheBlock location has invalid slot");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(parent.occupancy[slot] && parent.occupied_count > 0, "CacheBlock location is not occupied");
        GroupAvailability& group = group_availability_.at(*parent.bound_group);
        removeAvailability(group, parent, location.lcm_block_id);
        parent.occupancy[slot] = false;
        --parent.occupied_count;
        ++group.free_slots;
        if (parent.occupied_count == 0) {
            group.free_slots -= parent.occupancy.size();
            --group.bound_parents;
            parent.bound_group.reset();
            parent.occupancy.clear();
            parent.first_free_slots.clear();
            FatalCheck(free_parent_ids_.size() < lcm_blocks_.size(), "LCM free queue overflow");
            free_parent_ids_.push_back(location.lcm_block_id);
        } else {
            // A full parent has no index entries, so a bucket-count change
            // may have left its (all-empty) minima at the old width.
            const std::size_t buckets = group.parents_by_bucket.size();
            if (parent.first_free_slots.size() != buckets) {
                parent.first_free_slots.assign(buckets, group.packing);
            }
            auto& first = parent.first_free_slots[slot % buckets];
            first = std::min(first, location.slot_index);
            addAvailability(group, parent, location.lcm_block_id);
        }
    }

private:
    using ParentOrder = std::pair<std::int32_t, std::int32_t>;  // -occupancy, parent ID
    struct GroupAvailability {
        std::int32_t packing{0};
        std::size_t free_slots{0};  // Holes in bound parents; empty parents remain pool-wide.
        std::size_t bound_parents{0};
        std::vector<std::set<ParentOrder>> parents_by_bucket;
    };
    struct LcmBlock {
        std::optional<std::uint32_t> bound_group;
        std::vector<bool> occupancy;
        std::uint32_t occupied_count{0};
        // Only the lowest free slot per bucket is needed, not a list of all holes.
        // The packing is the sentinel for a bucket with no free slot.
        std::vector<std::int32_t> first_free_slots;
    };

    std::size_t availableBlocks(std::uint32_t group_id, std::int32_t packing) const {
        const auto it = group_availability_.find(group_id);
        if (it != group_availability_.end()) {
            _assert(it->second.bound_parents == 0 || it->second.packing == packing,
                    "group packing changed while LCM block is occupied");
        }
        const std::size_t holes = it == group_availability_.end() ? 0 : it->second.free_slots;
        return holes + free_parent_ids_.size() * static_cast<std::size_t>(packing);
    }

    static void removeAvailability(GroupAvailability& group, const LcmBlock& parent, std::int32_t id) {
        if (parent.occupied_count == parent.occupancy.size()) {
            return;
        }
        for (std::size_t bucket = 0; bucket < group.parents_by_bucket.size(); ++bucket) {
            if (parent.first_free_slots[bucket] < group.packing) {
                group.parents_by_bucket[bucket].erase({-static_cast<std::int32_t>(parent.occupied_count), id});
            }
        }
    }

    static void addAvailability(GroupAvailability& group, const LcmBlock& parent, std::int32_t id) {
        if (parent.occupied_count == parent.occupancy.size()) {
            return;
        }
        for (std::size_t bucket = 0; bucket < group.parents_by_bucket.size(); ++bucket) {
            if (parent.first_free_slots[bucket] < group.packing) {
                group.parents_by_bucket[bucket].emplace(-static_cast<std::int32_t>(parent.occupied_count), id);
            }
        }
    }

    GroupAvailability& prepareAvailability(std::uint32_t group_id, std::int32_t packing,
                                           std::size_t requested_buckets = 0) {
        (void)availableBlocks(group_id, packing);  // Validate a live group's packing before mutation.
        auto [it, inserted] = group_availability_.try_emplace(group_id);
        GroupAvailability& group = it->second;
        if (inserted || (group.bound_parents == 0 && group.packing != packing)) {
            group = GroupAvailability{.packing = packing};
            group.parents_by_bucket.resize(requested_buckets > 0 ? requested_buckets : 1);
        }
        if (requested_buckets == 0 || requested_buckets == group.parents_by_bucket.size()) {
            return group;
        }
        // Mixed ordinary/balanced callers are allowed. Configure on the first
        // balanced acquire (or an explicit change), visiting only partial
        // parents of this group. Steady-state allocation never rebuilds this.
        std::vector<std::int32_t> partial_parents;
        for (const auto& bucket : group.parents_by_bucket) {
            for (const auto& [occupied, id] : bucket) {
                partial_parents.push_back(id);
            }
        }
        std::ranges::sort(partial_parents);
        partial_parents.erase(std::unique(partial_parents.begin(), partial_parents.end()), partial_parents.end());
        group.parents_by_bucket.clear();
        group.parents_by_bucket.resize(requested_buckets);
        for (std::int32_t id : partial_parents) {
            LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(id - 1)];
            parent.first_free_slots.assign(requested_buckets, packing);
            for (std::size_t slot = 0; slot < parent.occupancy.size(); ++slot) {
                if (!parent.occupancy[slot]) {
                    auto& first = parent.first_free_slots[slot % requested_buckets];
                    first = std::min(first, static_cast<std::int32_t>(slot));
                }
            }
            addAvailability(group, parent, id);
        }
        return group;
    }

    CacheBlockLocation nextLocation(const GroupAvailability& group, std::span<const std::int64_t> loads) const {
        // Ordinary allocation orders by parent occupancy, parent ID, then slot.
        // Balanced allocation puts request load and bucket ID before that order.
        using Priority = std::tuple<std::int64_t, std::size_t, std::int32_t, std::int32_t, std::int32_t>;
        std::optional<Priority> best;
        CacheBlockLocation location;
        for (std::size_t bucket = 0; bucket < group.parents_by_bucket.size(); ++bucket) {
            if (group.parents_by_bucket[bucket].empty()) {
                continue;
            }
            const auto [occupied, id] = *group.parents_by_bucket[bucket].begin();
            const auto slot = lcmBlock(id).first_free_slots[bucket];
            const Priority priority{loads.empty() ? 0 : loads[bucket], loads.empty() ? 0 : bucket, occupied, id, slot};
            if (!best || priority < *best) {
                best = priority;
                location = CacheBlockLocation{id, slot};
            }
        }
        if (best) {
            return location;
        }
        // Every hole in bound parents is exhausted before opening a FIFO parent.
        FatalCheck(!free_parent_ids_.empty(), "prechecked block capacity was exhausted");
        const auto slot =
            loads.empty() ? 0 : static_cast<std::int32_t>(std::ranges::min_element(loads) - loads.begin());
        return CacheBlockLocation{free_parent_ids_.front(), slot};
    }

    std::vector<CacheBlockRef> acquireAvailableBlocks(std::uint32_t group_id, std::int32_t packing, std::int32_t count,
                                                      std::span<const std::int32_t> bucket_loads = {}) {
        const bool balanced = bucket_loads.size() > 1;
        GroupAvailability& group = prepareAvailability(group_id, packing, balanced ? bucket_loads.size() : 0);
        std::vector<std::int64_t> loads;
        if (balanced) {
            loads.assign(bucket_loads.begin(), bucket_loads.end());
        }
        // Capacity-first selection can consume every compatible hole, then
        // every empty parent. The exact capacity check precedes all mutations,
        // so a second, pool-sized shadow allocator is unnecessary.
        std::vector<CacheBlockRef> out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::int32_t i = 0; i < count; ++i) {
            const CacheBlockLocation location = nextLocation(group, loads);
            out.push_back(createBlockRef(group_id, packing, location));
            if (balanced) {
                ++loads[static_cast<std::size_t>(location.slot_index) % loads.size()];
            }
        }
        return out;
    }

    static std::size_t checkedLcmBlockCount(std::int32_t num_lcm_blocks) {
        _assert(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        return static_cast<std::size_t>(num_lcm_blocks);
    }

    const LcmBlock& lcmBlock(std::int32_t lcm_block_id) const {
        _assert(lcm_block_id > 0 && static_cast<std::size_t>(lcm_block_id) <= lcm_blocks_.size(),
                "LCM block id out of range");
        return lcm_blocks_[static_cast<std::size_t>(lcm_block_id - 1)];
    }

    CacheBlockRef createBlockRef(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) {
        auto* control = new internal_cache_block_ref::CacheBlockControl(*this, location);
        // Allocate the control before mutating the pool, then commit the
        // location before publishing its RAII owner: CacheBlock destruction
        // releases this location and therefore requires it to be occupied.
        occupy(group_id, slots_per_parent, location);
        return CacheBlockRef{*control};
    }

    void occupy(std::uint32_t group_id, std::int32_t slots_per_parent, CacheBlockLocation location) noexcept {
        LcmBlock& parent = lcm_blocks_[static_cast<std::size_t>(location.lcm_block_id - 1)];
        GroupAvailability& group = group_availability_.at(group_id);
        const std::size_t buckets = group.parents_by_bucket.size();
        if (parent.occupied_count == 0) {
            FatalCheck(!free_parent_ids_.empty() && free_parent_ids_.front() == location.lcm_block_id,
                       "empty LCM placement must consume the next free parent");
            FatalCheck(parent.occupancy.empty(), "empty LCM parent must not retain child slots");
            parent.occupancy.assign(static_cast<std::size_t>(slots_per_parent), false);
            if (slots_per_parent > 1) {
                parent.first_free_slots.resize(buckets);
                for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
                    parent.first_free_slots[bucket] = static_cast<std::int32_t>(bucket);
                }
            }
            free_parent_ids_.pop_front();
            parent.bound_group = group_id;
            ++group.bound_parents;
            group.free_slots += static_cast<std::size_t>(slots_per_parent);
        } else {
            removeAvailability(group, parent, location.lcm_block_id);
        }
        FatalCheck(
            parent.bound_group == group_id && parent.occupancy.size() == static_cast<std::size_t>(slots_per_parent),
            "LCM parent binding changed while occupied");
        const std::size_t slot = static_cast<std::size_t>(location.slot_index);
        FatalCheck(slot < parent.occupancy.size(), "LCM child slot is out of range");
        FatalCheck(!parent.occupancy[slot], "LCM child slot already occupied");
        parent.occupancy[slot] = true;
        ++parent.occupied_count;
        --group.free_slots;
        if (slots_per_parent > 1) {
            auto& first = parent.first_free_slots[slot % buckets];
            if (first == location.slot_index) {
                first = slots_per_parent;
                for (std::size_t next = slot + buckets; next < parent.occupancy.size(); next += buckets) {
                    if (!parent.occupancy[next]) {
                        first = static_cast<std::int32_t>(next);
                        break;
                    }
                }
            }
        }
        addAvailability(group, parent, location.lcm_block_id);
    }

    std::vector<LcmBlock> lcm_blocks_;
    // Free parents are interchangeable: release appends and allocation consumes
    // the front. Bound parents have per-group, per-bucket availability indices.
    std::deque<std::int32_t> free_parent_ids_;
    std::map<std::uint32_t, GroupAvailability> group_availability_;
};

}  // namespace tokenspeed
