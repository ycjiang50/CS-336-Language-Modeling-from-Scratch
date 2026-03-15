"""
KV cache pool and radix prefix cache.

Design overview
───────────────
KVPool
    Pre-allocated GPU memory organized as a flat pool of *slots*.
    Each slot stores key/value vectors for **one token position** across all
    layers.  Requests are allocated contiguous runs of slots; after a request
    finishes its slots are either donated to the RadixPrefixCache or freed.

    Pool layout per layer:
        k_cache[layer] : (pool_size, num_heads, head_dim)
        v_cache[layer] : (pool_size, num_heads, head_dim)

    The gather/scatter helpers convert between pool storage and the model's
    native ``past_key_values`` format ``List[(1, heads, seq, head_dim)]``.

    In a production system (SGLang / vLLM) the attention kernel reads
    directly from the paged pool via a page-table indirection; here we
    copy-gather for compatibility with the unmodified TransformerLM.

RadixPrefixCache
    A trie (radix tree) keyed by token-ID sequences.  Each node maps to
    one pool slot.  When a new request arrives the scheduler walks the tree
    to find the longest matching prefix, *locks* those nodes (preventing
    eviction), and reuses the corresponding KV.  Eviction is LRU on
    unreferenced leaf nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════
# KV Cache Pool
# ═══════════════════════════════════════════════════════════════════════

class KVPool:
    """Slot-based KV cache pool backed by pre-allocated tensors."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        pool_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pool_size = pool_size
        self.device = device
        self.dtype = dtype

        self.k_cache = [
            torch.zeros(pool_size, num_heads, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(pool_size, num_heads, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self._free_slots: List[int] = list(range(pool_size - 1, -1, -1))
        self._ref_count: List[int] = [0] * pool_size

    # ── capacity ──────────────────────────────────────────────────────

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    # ── allocation ────────────────────────────────────────────────────

    def allocate(self, n: int) -> Optional[List[int]]:
        """Allocate *n* slots. Returns slot indices or ``None`` if OOM."""
        if len(self._free_slots) < n:
            return None
        slots = [self._free_slots.pop() for _ in range(n)]
        for s in slots:
            self._ref_count[s] = 1
        return slots

    def free(self, slots: List[int]) -> None:
        """Decrement ref-count; return to free-list when it reaches zero."""
        for s in slots:
            self._ref_count[s] -= 1
            if self._ref_count[s] <= 0:
                self._ref_count[s] = 0
                self._free_slots.append(s)

    def inc_ref(self, slots: List[int]) -> None:
        for s in slots:
            self._ref_count[s] += 1

    # ── gather / scatter (pool ↔ model format) ────────────────────────

    def gather_kv(
        self, slots: List[int]
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Build ``past_key_values`` from pool slots.

        Returns
        -------
        List of ``(k, v)`` per layer, each shaped ``(1, heads, seq, head_dim)``,
        or ``None`` when *slots* is empty.
        """
        if not slots:
            return None
        idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        past_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.num_layers):
            k = self.k_cache[layer][idx]                        # (S, H, D)
            v = self.v_cache[layer][idx]
            k = k.unsqueeze(0).permute(0, 2, 1, 3)             # (1, H, S, D)
            v = v.unsqueeze(0).permute(0, 2, 1, 3)
            past_kv.append((k, v))
        return past_kv

    def scatter_kv(
        self,
        slots: List[int],
        full_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        offset: int,
    ) -> None:
        """
        Write *newly computed* KV entries back into the pool.

        Parameters
        ----------
        slots   : pool slots to write into (length N)
        full_kv : model-returned ``past_key_values`` (concatenated old+new)
        offset  : where the new entries start inside the full KV sequence
        """
        n = len(slots)
        if n == 0:
            return
        idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        for layer, (k_full, v_full) in enumerate(full_kv):
            # k_full: (1, H, total_len, D)
            k_new = k_full[0, :, offset:offset + n, :].permute(1, 0, 2)  # (N, H, D)
            v_new = v_full[0, :, offset:offset + n, :].permute(1, 0, 2)
            self.k_cache[layer][idx] = k_new
            self.v_cache[layer][idx] = v_new


# ═══════════════════════════════════════════════════════════════════════
# Radix Prefix Cache
# ═══════════════════════════════════════════════════════════════════════

class RadixNode:
    """Single node in the radix prefix tree."""

    __slots__ = ("children", "pool_slot", "ref_count", "last_access",
                 "parent", "token_id")

    def __init__(
        self,
        token_id: int = -1,
        pool_slot: int = -1,
        parent: Optional["RadixNode"] = None,
    ):
        self.children: Dict[int, RadixNode] = {}
        self.pool_slot = pool_slot
        self.ref_count = 0          # active locks (prevents eviction)
        self.last_access = time.time()
        self.parent = parent
        self.token_id = token_id


@dataclass
class CacheHandle:
    """Opaque handle returned by ``match_prefix``; used for lock / unlock."""

    nodes: List[RadixNode] = field(default_factory=list)
    matched_len: int = 0

    def slots(self) -> List[int]:
        return [n.pool_slot for n in self.nodes]


class RadixPrefixCache:
    """
    Radix-tree prefix cache for KV reuse across requests.

    Each path from root to a node encodes a unique token-ID prefix.
    Pool-slot ownership is shared: when a request finishes it *inserts*
    its full sequence into the tree (the tree inc-refs new slots).
    Eviction removes LRU leaf nodes with ``ref_count == 0``.
    """

    def __init__(self, pool: KVPool):
        self.root = RadixNode()
        self.pool = pool
        self._total_cached = 0

    @property
    def total_cached(self) -> int:
        return self._total_cached

    # ── lookup ────────────────────────────────────────────────────────

    def match_prefix(self, input_ids: List[int]) -> CacheHandle:
        """Walk the tree and return the longest matching prefix."""
        node = self.root
        matched: List[RadixNode] = []
        for tid in input_ids:
            child = node.children.get(tid)
            if child is None:
                break
            child.last_access = time.time()
            matched.append(child)
            node = child
        return CacheHandle(nodes=matched, matched_len=len(matched))

    # ── lock / unlock (ref-counting) ──────────────────────────────────

    def lock(self, handle: CacheHandle) -> None:
        """Prevent matched nodes from being evicted while a request uses them."""
        for node in handle.nodes:
            node.ref_count += 1
        self.pool.inc_ref(handle.slots())

    def unlock(self, handle: CacheHandle) -> None:
        """Release the lock taken by ``lock``."""
        for node in handle.nodes:
            node.ref_count -= 1
        self.pool.free(handle.slots())

    # ── insert ────────────────────────────────────────────────────────

    def insert(self, token_ids: List[int], pool_slots: List[int]) -> None:
        """
        Insert a token sequence and its pool slots into the cache.

        For tokens that already exist in the tree the node is left as-is
        (no ref-count change).  For new tokens a fresh node is created and
        the pool slot is inc-ref'd (the tree now co-owns that slot).
        """
        assert len(token_ids) == len(pool_slots)
        node = self.root
        for i, tid in enumerate(token_ids):
            child = node.children.get(tid)
            if child is not None:
                child.last_access = time.time()
                node = child
            else:
                new_node = RadixNode(
                    token_id=tid,
                    pool_slot=pool_slots[i],
                    parent=node,
                )
                new_node.last_access = time.time()
                node.children[tid] = new_node
                self.pool.inc_ref([pool_slots[i]])
                self._total_cached += 1
                node = new_node

    # ── eviction ──────────────────────────────────────────────────────

    def evict(self, num_slots: int) -> int:
        """
        Evict up to *num_slots* LRU leaf nodes.

        Returns the number of slots actually freed.
        """
        freed = 0
        candidates = self._eviction_candidates()
        for node in candidates:
            if freed >= num_slots:
                break
            if node.ref_count > 0 or node.children:
                continue
            self.pool.free([node.pool_slot])
            if node.parent is not None and node.token_id in node.parent.children:
                del node.parent.children[node.token_id]
            self._total_cached -= 1
            freed += 1
        return freed

    def _eviction_candidates(self) -> List[RadixNode]:
        """Collect leaf nodes sorted oldest-first (LRU order)."""
        leaves: List[RadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.children and node is not self.root:
                leaves.append(node)
            else:
                stack.extend(node.children.values())
        leaves.sort(key=lambda n: n.last_access)
        return leaves
