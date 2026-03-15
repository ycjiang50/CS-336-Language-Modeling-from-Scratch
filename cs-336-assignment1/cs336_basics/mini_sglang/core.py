"""Core data structures for the mini-sglang serving engine."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class RequestStatus(Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    """Parameters controlling token sampling during generation."""

    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1
    max_new_tokens: int = 256
    stop_token_ids: List[int] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class StreamChunk:
    """One chunk of a streaming response pushed to the output queue."""

    token_id: int
    finished: bool
    finish_reason: Optional[str] = None


@dataclass
class Request:
    """
    Represents a single inference request throughout its lifecycle.

    Lifecycle: PENDING → PREFILLING → DECODING → FINISHED / ABORTED

    The request tracks both the token IDs and the KV-cache pool slots
    that back them, enabling the scheduler to manage memory efficiently.
    """

    uid: int
    input_ids: List[int]
    sampling_params: SamplingParams
    status: RequestStatus = RequestStatus.PENDING
    output_ids: List[int] = field(default_factory=list)

    # ── KV cache bookkeeping ──────────────────────────────────────────
    pool_slots: List[int] = field(default_factory=list)
    cached_len: int = 0       # tokens reused from radix prefix cache
    prefill_len: int = 0      # tokens computed during prefill

    cache_handle: Any = None  # RadixPrefixCache handle for locked prefix

    # ── Timing ────────────────────────────────────────────────────────
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    # ── Async streaming output ────────────────────────────────────────
    output_queue: Optional[asyncio.Queue] = None

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def seq_len(self) -> int:
        return len(self.input_ids) + len(self.output_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)


@dataclass
class SchedulerStats:
    """Live statistics exposed by the scheduler."""

    pending_requests: int = 0
    running_requests: int = 0
    total_requests: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    total_cache_hit_tokens: int = 0
    kv_pool_free: int = 0
    kv_pool_total: int = 0
    prefix_cache_entries: int = 0
