# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CachePlacement
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)


def cache_debug_enabled() -> bool:
    """Whether expensive, GPU-synchronizing cache validation is enabled."""
    return os.environ.get("TOKENSPEED_CACHE_DEBUG") == "1"


def require_positive_int(name: str, value: object) -> int:
    """Validate that ``value`` is a positive, non-boolean integer.

    Args:
        name: Field name used in the error message.
        value: Value to validate.

    Returns:
        ``value`` unchanged, typed as ``int``.

    Raises:
        ValueError: If ``value`` is a bool, not an int, or not positive.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class CacheGroupAddressSpace:
    """One group's physical storage and derived scheduler address space.

    ``physical_packing`` and ``num_lcm_blocks`` come from the local plan;
    ``placement`` comes from its declaration. DCP topology is shared across
    groups. The scheduler only consumes the derived integer bucket count.
    """

    physical_packing: int
    num_lcm_blocks: int
    placement: CachePlacement = "replicated"
    dcp_size: int = 1

    def __post_init__(self) -> None:
        require_positive_int("physical_packing", self.physical_packing)
        require_positive_int("num_lcm_blocks", self.num_lcm_blocks)
        require_positive_int("dcp_size", self.dcp_size)
        if self.placement not in ("replicated", "virtual_block_cyclic"):
            raise ValueError(f"unsupported cache placement {self.placement!r}")
        if self.virtual_block_count - 1 > (1 << 31) - 1:
            raise ValueError("virtual cache block ID exceeds int32 range")

    @property
    def allocation_bucket_count(self) -> int:
        return self.dcp_size if self.placement == "virtual_block_cyclic" else 1

    @property
    def local_page_count(self) -> int:
        return 1 + self.num_lcm_blocks * self.physical_packing

    @property
    def virtual_packing(self) -> int:
        return self.physical_packing * self.allocation_bucket_count

    @property
    def virtual_block_count(self) -> int:
        return 1 + self.num_lcm_blocks * self.virtual_packing

    def local_block(self, virtual_block: int, rank: int) -> tuple[int, bool]:
        """Return a safe local page and ownership; null/nonowners map to 0.

        Args:
            virtual_block: Scheduler block ID, including reserved null ID 0.
            rank: This process's rank in the DCP subgroup.

        Returns:
            ``(local_page, valid_owner)``. A false mask forbids stores.
        """
        if not 0 <= rank < self.dcp_size:
            raise ValueError("DCP rank is out of range")
        if not 0 <= virtual_block < self.virtual_block_count:
            raise ValueError("virtual cache block ID is out of range")
        if virtual_block == 0:
            return 0, False
        buckets = self.allocation_bucket_count
        if buckets > 1 and (virtual_block - 1) % buckets != rank:
            return 0, False
        return (virtual_block - 1) // buckets + 1, True

    def virtual_block(self, local_page: int, rank: int) -> int:
        """Return the scheduler ID owned by ``rank`` at a local page.

        Null page 0 remains null. Replicated pages have the same ID on all
        ranks. Nonnull pages must fit the local plan.
        """
        if not 0 <= rank < self.dcp_size:
            raise ValueError("DCP rank is out of range")
        if not 0 <= local_page < self.local_page_count:
            raise ValueError("local cache page ID is out of range")
        if local_page == 0:
            return 0
        buckets = self.allocation_bucket_count
        return (local_page - 1) * buckets + (rank if buckets > 1 else 0) + 1


@dataclass(frozen=True)
class CacheRuntimeContract:
    prefix_granularity: int
    num_lcm_blocks: int
    token_capacity: int
    group_specs: tuple[CacheGroupSpec, ...]
    # Both projected from the memory plan, which owns physical geometry: how
    # many CacheBlocks each group has, and how many share one LCM parent.
    group_page_counts: Mapping[str, int]
    group_packing: Mapping[str, int]
    group_placements: Mapping[str, CachePlacement] = field(default_factory=dict)
    dcp_size: int = 1
    group_address_spaces: Mapping[str, CacheGroupAddressSpace] = field(init=False)

    def __post_init__(self) -> None:
        prefix_granularity = require_positive_int(
            "prefix_granularity", self.prefix_granularity
        )
        num_lcm_blocks = require_positive_int("num_lcm_blocks", self.num_lcm_blocks)
        token_capacity = require_positive_int("token_capacity", self.token_capacity)
        require_positive_int("dcp_size", self.dcp_size)
        if not isinstance(self.group_specs, tuple) or not self.group_specs:
            raise ValueError("group_specs must be a non-empty tuple")
        if any(not isinstance(spec, CacheGroupSpec) for spec in self.group_specs):
            raise ValueError("group_specs must contain CacheGroupSpec values")
        group_ids = tuple(spec.group_id for spec in self.group_specs)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("group_specs contain duplicate group IDs")
        counts = dict(self.group_page_counts)
        actual_group_ids = set(counts)
        expected_group_ids = set(group_ids)
        if actual_group_ids != expected_group_ids:
            raise ValueError(
                "group_page_counts keys must match group_specs: "
                f"missing={sorted(expected_group_ids - actual_group_ids)} "
                f"extra={sorted(actual_group_ids - expected_group_ids)}"
            )
        counts = {
            group_id: require_positive_int(
                f"group page count for {group_id!r}", counts[group_id]
            )
            for group_id in group_ids
        }
        packing = dict(self.group_packing)
        if set(packing) != expected_group_ids:
            raise ValueError(
                "group_packing keys must match group_specs: "
                f"missing={sorted(expected_group_ids - set(packing))} "
                f"extra={sorted(set(packing) - expected_group_ids)}"
            )
        packing = {
            group_id: require_positive_int(
                f"cache_blocks_per_lcm_block for {group_id!r}", packing[group_id]
            )
            for group_id in group_ids
        }
        # Both sides come from the plan, so this checks the plan's own
        # arithmetic: every group's blocks are its packing per parent times
        # the parent count, plus the reserved null block.
        expected_counts = {
            group_id: num_lcm_blocks * packing[group_id] + 1 for group_id in group_ids
        }
        if counts != expected_counts:
            raise ValueError(
                "group page counts must equal num_lcm_blocks * "
                "cache_blocks_per_lcm_block + 1: "
                f"expected={expected_counts}, got={counts}"
            )
        placements = dict(self.group_placements) or dict.fromkeys(
            group_ids, "replicated"
        )
        if set(placements) != expected_group_ids:
            raise ValueError("group_placements keys must match group_specs")
        spaces = {
            group_id: CacheGroupAddressSpace(
                packing[group_id], num_lcm_blocks, placements[group_id], self.dcp_size
            )
            for group_id in group_ids
        }
        for spec in self.group_specs:
            if spec.rows_per_page is not None:
                max_slots = spaces[spec.group_id].local_page_count * spec.rows_per_page
                if max_slots - 1 > (1 << 31) - 1:
                    raise ValueError(
                        f"local cache slots for {spec.group_id!r} exceed int32 range"
                    )
        max_child_pages = (
            max(space.virtual_block_count for space in spaces.values()) - 1
        )
        if token_capacity > max_child_pages * prefix_granularity:
            raise ValueError(
                "token_capacity exceeds the largest group's child-page capacity"
            )
        object.__setattr__(self, "group_page_counts", MappingProxyType(counts))
        object.__setattr__(self, "group_packing", MappingProxyType(packing))
        object.__setattr__(self, "group_placements", MappingProxyType(placements))
        object.__setattr__(self, "group_address_spaces", MappingProxyType(spaces))

    @property
    def virtual_block_counts(self) -> Mapping[str, int]:
        """Scheduler block counts, including the null block."""
        return {
            key: space.virtual_block_count
            for key, space in self.group_address_spaces.items()
        }

    @property
    def virtual_packing(self) -> Mapping[str, int]:
        """Scheduler children per parent; physical binding uses group_packing."""
        return {
            key: space.virtual_packing
            for key, space in self.group_address_spaces.items()
        }
