"""
Continuous-batching scheduler.

The scheduler owns the lifecycle of every request:

    PENDING  →  PREFILLING  →  DECODING  →  FINISHED / ABORTED

Main loop (``run``):
    1. Drain the async input queue for new / abort messages.
    2. Try to prefill one pending request (prefix-cache lookup → allocate
       KV slots → model forward → sample first token).
    3. Run one decode step for *all* running requests (model forward per
       request → sample next token → check termination).
    4. Yield to the event loop so the HTTP server can send tokens.

Continuous batching comes from the fact that steps 2 and 3 are interleaved:
a new request can be admitted *between* decode iterations of existing
requests, so no request blocks the pipeline.

Note: the current engine runs forward passes one-request-at-a-time because
the underlying TransformerLM stores KV as per-request ``List[(B, H, S, D)]``
tensors.  A production system (SGLang / vLLM) uses paged-attention kernels
that read directly from the pool via a page-table, enabling true GPU-level
batching across requests with different KV lengths.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Union

import torch

from .core import Request, RequestStatus, SamplingParams, SchedulerStats, StreamChunk
from .engine import ModelEngine, sample_token
from .kv_cache import KVPool, RadixPrefixCache

logger = logging.getLogger("mini_sglang.scheduler")


class Scheduler:
    """Continuous-batching scheduler with prefix-cache support."""

    def __init__(
        self,
        engine: ModelEngine,
        kv_pool: KVPool,
        prefix_cache: RadixPrefixCache,
        max_running: int = 64,
        max_context_len: int = 256,
    ):
        self.engine = engine
        self.kv_pool = kv_pool
        self.prefix_cache = prefix_cache
        self.max_running = max_running
        self.max_context_len = max_context_len

        self.pending: List[Request] = []
        self.running: Dict[int, Request] = {}  # uid → Request

        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

        # cumulative stats
        self._total_requests = 0
        self._total_prefill_tokens = 0
        self._total_decode_tokens = 0
        self._total_cache_hits = 0

    # ── public API (called from the HTTP layer) ───────────────────────

    async def add_request(self, req: Request) -> None:
        await self._input_queue.put(req)

    async def abort_request(self, uid: int) -> None:
        await self._input_queue.put(("abort", uid))

    def get_stats(self) -> SchedulerStats:
        return SchedulerStats(
            pending_requests=len(self.pending),
            running_requests=len(self.running),
            total_requests=self._total_requests,
            total_prefill_tokens=self._total_prefill_tokens,
            total_decode_tokens=self._total_decode_tokens,
            total_cache_hit_tokens=self._total_cache_hits,
            kv_pool_free=self.kv_pool.num_free,
            kv_pool_total=self.kv_pool.pool_size,
            prefix_cache_entries=self.prefix_cache.total_cached,
        )

    # ── main loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Scheduler main loop – runs as an ``asyncio.Task``."""
        self._running = True
        logger.info("Scheduler started")
        while self._running:
            self._drain_input_queue()

            had_work = False

            # Admit one pending request per iteration (fairness with decode)
            if self._try_prefill():
                had_work = True

            # One decode step for all running requests
            if self.running:
                self._decode_step()
                had_work = True

            if had_work:
                await asyncio.sleep(0)        # yield to event loop
            else:
                await asyncio.sleep(0.005)    # idle – back off

    def stop(self) -> None:
        self._running = False

    # ── internal: input handling ──────────────────────────────────────

    def _drain_input_queue(self) -> None:
        while not self._input_queue.empty():
            try:
                item = self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if isinstance(item, tuple) and item[0] == "abort":
                self._handle_abort(item[1])
            elif isinstance(item, Request):
                self.pending.append(item)
                self._total_requests += 1

    def _handle_abort(self, uid: int) -> None:
        self.pending = [r for r in self.pending if r.uid != uid]
        if uid in self.running:
            req = self.running.pop(uid)
            self._release_resources(req)
            req.status = RequestStatus.ABORTED
            if req.output_queue:
                req.output_queue.put_nowait(
                    StreamChunk(token_id=-1, finished=True, finish_reason="abort")
                )

    # ── internal: prefill ─────────────────────────────────────────────

    def _try_prefill(self) -> bool:
        """Prefill one pending request.  Returns True on success."""
        if not self.pending:
            return False
        if len(self.running) >= self.max_running:
            return False

        req = self.pending[0]

        # 1. Prefix cache lookup
        handle = self.prefix_cache.match_prefix(req.input_ids)
        cached_len = handle.matched_len

        if cached_len > 0:
            self.prefix_cache.lock(handle)
            req.cache_handle = handle
            req.cached_len = cached_len
            req.pool_slots = list(handle.slots())
            self._total_cache_hits += cached_len
            logger.info(
                "req %d: prefix hit %d/%d tokens", req.uid, cached_len, req.prompt_len
            )

        extend_len = req.prompt_len - cached_len

        # 2. Allocate pool slots for un-cached prompt tokens
        new_slots = self._allocate_slots(extend_len)
        if new_slots is None:
            # Undo the lock we just took
            if req.cache_handle:
                self.prefix_cache.unlock(req.cache_handle)
                req.cache_handle = None
                req.cached_len = 0
                req.pool_slots = []
            logger.warning("req %d: KV pool exhausted, deferring", req.uid)
            return False

        self.pending.pop(0)
        req.status = RequestStatus.PREFILLING

        # 3. Forward pass on the un-cached suffix
        uncached_ids = req.input_ids[cached_len:]
        input_t = torch.tensor([uncached_ids], dtype=torch.long, device=self.engine.device)

        past_kv = self.kv_pool.gather_kv(req.pool_slots[:cached_len]) if cached_len > 0 else None
        logits, new_kv = self.engine.forward(input_t, past_key_values=past_kv)

        # 4. Scatter new KV back to pool
        self.kv_pool.scatter_kv(new_slots, new_kv, offset=cached_len)
        req.pool_slots = req.pool_slots[:cached_len] + new_slots
        req.prefill_len = extend_len
        self._total_prefill_tokens += extend_len

        # 5. Sample first token
        token = sample_token(logits[0], req.sampling_params, req.input_ids)
        req.output_ids.append(token)
        req.first_token_time = time.time()

        # 6. Stream + transition
        finished = self._check_finished(req)
        if req.output_queue:
            req.output_queue.put_nowait(StreamChunk(
                token_id=token,
                finished=finished,
                finish_reason="stop" if finished else None,
            ))

        if finished:
            req.status = RequestStatus.FINISHED
            req.finish_time = time.time()
            self._release_resources(req)
        else:
            req.status = RequestStatus.DECODING
            self.running[req.uid] = req

        return True

    # ── internal: decode ──────────────────────────────────────────────

    def _decode_step(self) -> None:
        """Run one decode iteration for every running request."""
        finished_uids: List[int] = []

        for uid, req in list(self.running.items()):
            new_slot = self._allocate_slots(1)
            if new_slot is None:
                logger.warning("req %d: no KV slot for decode, skipping step", uid)
                continue

            last_token = req.output_ids[-1]
            input_t = torch.tensor(
                [[last_token]], dtype=torch.long, device=self.engine.device
            )

            past_kv = self.kv_pool.gather_kv(req.pool_slots)
            logits, new_kv = self.engine.forward(input_t, past_key_values=past_kv)

            self.kv_pool.scatter_kv(new_slot, new_kv, offset=len(req.pool_slots))
            req.pool_slots.extend(new_slot)
            self._total_decode_tokens += 1

            all_tokens = req.input_ids + req.output_ids
            token = sample_token(logits[0], req.sampling_params, all_tokens)
            req.output_ids.append(token)

            finished = self._check_finished(req)
            if req.output_queue:
                req.output_queue.put_nowait(StreamChunk(
                    token_id=token,
                    finished=finished,
                    finish_reason="stop" if finished else None,
                ))

            if finished:
                finished_uids.append(uid)

        for uid in finished_uids:
            req = self.running.pop(uid)
            req.status = RequestStatus.FINISHED
            req.finish_time = time.time()
            self._release_resources(req)

    # ── internal: resource management ─────────────────────────────────

    def _allocate_slots(self, n: int) -> Optional[List[int]]:
        """Allocate *n* pool slots, attempting eviction on failure."""
        slots = self.kv_pool.allocate(n)
        if slots is not None:
            return slots
        needed = n - self.kv_pool.num_free
        freed = self.prefix_cache.evict(needed)
        if freed < needed:
            return None
        return self.kv_pool.allocate(n)

    def _release_resources(self, req: Request) -> None:
        """
        Donate the request's KV to the prefix cache and release refs.

        Pool-slot ownership transfer:
            - Slots that the request *borrowed* from the tree (via prefix
              match) are unlocked (dec-ref); the tree retains its ref.
            - Slots that the request *allocated* (prefill + decode) are
              inserted into the tree (inc-ref) and then freed (dec-ref),
              leaving the tree as sole owner.
        """
        cacheable_ids = (req.input_ids + req.output_ids)[: len(req.pool_slots)]
        if req.pool_slots:
            self.prefix_cache.insert(cacheable_ids, req.pool_slots)

        if req.cache_handle:
            self.prefix_cache.unlock(req.cache_handle)
            req.cache_handle = None

        owned_slots = req.pool_slots[req.cached_len:]
        if owned_slots:
            self.kv_pool.free(owned_slots)

        req.pool_slots = []

    # ── internal: termination check ───────────────────────────────────

    def _check_finished(self, req: Request) -> bool:
        if len(req.output_ids) >= req.sampling_params.max_new_tokens:
            return True
        if req.output_ids and req.output_ids[-1] in req.sampling_params.stop_token_ids:
            return True
        if req.seq_len >= self.max_context_len:
            return True
        return False
