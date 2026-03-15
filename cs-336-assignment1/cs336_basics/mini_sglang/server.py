"""
FastAPI HTTP server with OpenAI-compatible and custom endpoints.

Endpoints
─────────
GET  /health            Health check
GET  /v1/models         List available models
GET  /stats             Live scheduler / KV-pool statistics
POST /generate          Simple generate (+ SSE streaming)
POST /v1/completions    OpenAI-compatible completions (+ SSE streaming)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .core import Request, SamplingParams, StreamChunk
from .scheduler import Scheduler

logger = logging.getLogger("mini_sglang.server")


# ═══════════════════════════════════════════════════════════════════════
# Pydantic models for the HTTP API
# ═══════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = -1
    stream: bool = False
    stop_token_ids: Optional[List[int]] = None
    repetition_penalty: float = 1.0


class CompletionRequest(BaseModel):
    model: str = "transformer-lm"
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None
    repetition_penalty: float = 1.0


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str]


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# ═══════════════════════════════════════════════════════════════════════
# Shared server state
# ═══════════════════════════════════════════════════════════════════════

class ServerState:
    """Mutable state shared between the FastAPI routes and the scheduler."""

    def __init__(self, scheduler: Scheduler, tokenizer):
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self._uid_counter = 0

    def next_uid(self) -> int:
        self._uid_counter += 1
        return self._uid_counter

    def get_stop_ids(self, extra: Optional[List[int]] = None) -> List[int]:
        ids: List[int] = list(extra or [])
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            ids.append(eos)
        return list(set(ids))


# ═══════════════════════════════════════════════════════════════════════
# SSE streaming helpers
# ═══════════════════════════════════════════════════════════════════════

async def _stream_generate(
    request: Request, tokenizer
) -> AsyncGenerator[str, None]:
    """Yield SSE events for ``/generate`` with incremental detokenization."""
    tokens: List[int] = []
    prev_text = ""
    while True:
        chunk: StreamChunk = await request.output_queue.get()
        if chunk.token_id >= 0:
            tokens.append(chunk.token_id)
        full_text = tokenizer.decode(tokens)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        payload = json.dumps({
            "text": delta,
            "token_id": chunk.token_id,
            "finished": chunk.finished,
        })
        yield f"data: {payload}\n\n"
        if chunk.finished:
            yield "data: [DONE]\n\n"
            break


async def _stream_completions(
    request: Request, tokenizer, model_name: str
) -> AsyncGenerator[str, None]:
    """Yield SSE events for ``/v1/completions``."""
    tokens: List[int] = []
    prev_text = ""
    while True:
        chunk: StreamChunk = await request.output_queue.get()
        if chunk.token_id >= 0:
            tokens.append(chunk.token_id)
        full_text = tokenizer.decode(tokens)
        delta = full_text[len(prev_text):]
        prev_text = full_text
        payload = {
            "id": f"cmpl-{request.uid}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "text": delta, "finish_reason": chunk.finish_reason}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        if chunk.finished:
            yield "data: [DONE]\n\n"
            break


# ═══════════════════════════════════════════════════════════════════════
# Collect all tokens (non-streaming helper)
# ═══════════════════════════════════════════════════════════════════════

async def _collect_tokens(output_queue: asyncio.Queue) -> List[int]:
    tokens: List[int] = []
    while True:
        chunk: StreamChunk = await output_queue.get()
        if chunk.token_id >= 0:
            tokens.append(chunk.token_id)
        if chunk.finished:
            break
    return tokens


# ═══════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════

def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(
        title="mini-sglang",
        description="Lightweight LLM serving engine with KV reuse and continuous batching",
    )

    # ── health & meta ─────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": "transformer-lm", "object": "model"}],
        }

    @app.get("/stats")
    async def stats():
        s = state.scheduler.get_stats()
        return {
            "pending_requests": s.pending_requests,
            "running_requests": s.running_requests,
            "total_requests": s.total_requests,
            "total_prefill_tokens": s.total_prefill_tokens,
            "total_decode_tokens": s.total_decode_tokens,
            "total_cache_hit_tokens": s.total_cache_hit_tokens,
            "kv_pool_free": s.kv_pool_free,
            "kv_pool_total": s.kv_pool_total,
            "prefix_cache_entries": s.prefix_cache_entries,
        }

    # ── /generate ─────────────────────────────────────────────────────

    @app.post("/generate")
    async def generate(body: GenerateRequest):
        uid = state.next_uid()
        input_ids = state.tokenizer.encode(body.prompt)

        sampling = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            max_new_tokens=body.max_new_tokens,
            stop_token_ids=state.get_stop_ids(body.stop_token_ids),
            repetition_penalty=body.repetition_penalty,
        )

        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        req = Request(
            uid=uid,
            input_ids=input_ids,
            sampling_params=sampling,
            output_queue=output_queue,
        )
        await state.scheduler.add_request(req)

        if body.stream:
            return StreamingResponse(
                _stream_generate(req, state.tokenizer),
                media_type="text/event-stream",
            )

        tokens = await _collect_tokens(output_queue)
        text = state.tokenizer.decode(tokens)
        ttft = (
            round((req.first_token_time - req.arrival_time) * 1000, 2)
            if req.first_token_time else None
        )
        total_ms = round((time.time() - req.arrival_time) * 1000, 2)

        return {
            "text": text,
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(input_ids) + len(tokens),
            "finish_reason": "stop",
            "time_to_first_token_ms": ttft,
            "total_time_ms": total_ms,
        }

    # ── /v1/completions (OpenAI-compatible) ───────────────────────────

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        uid = state.next_uid()
        input_ids = state.tokenizer.encode(body.prompt)

        sampling = SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            max_new_tokens=body.max_tokens,
            stop_token_ids=state.get_stop_ids(),
            repetition_penalty=body.repetition_penalty,
        )

        output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()
        req = Request(
            uid=uid,
            input_ids=input_ids,
            sampling_params=sampling,
            output_queue=output_queue,
        )
        await state.scheduler.add_request(req)

        if body.stream:
            return StreamingResponse(
                _stream_completions(req, state.tokenizer, body.model),
                media_type="text/event-stream",
            )

        tokens = await _collect_tokens(output_queue)
        text = state.tokenizer.decode(tokens)

        return CompletionResponse(
            id=f"cmpl-{uid}",
            created=int(time.time()),
            model=body.model,
            choices=[CompletionChoice(index=0, text=text, finish_reason="stop")],
            usage=CompletionUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(tokens),
                total_tokens=len(input_ids) + len(tokens),
            ),
        )

    return app
