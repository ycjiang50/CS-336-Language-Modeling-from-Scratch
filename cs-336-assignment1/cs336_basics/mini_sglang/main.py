#!/usr/bin/env python3
"""
Entry point for the mini-sglang serving engine.

Usage
─────
  # HTTP server mode (default)
  python -m cs336_basics.mini_sglang.main --checkpoint ckpt.pt

  # Interactive CLI mode (no HTTP server)
  python -m cs336_basics.mini_sglang.main --checkpoint ckpt.pt --interactive

  # Quick throughput benchmark
  python -m cs336_basics.mini_sglang.main --checkpoint ckpt.pt --benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import torch

# Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cs336_basics.layer import TransformerLM
from cs336_basics.mini_sglang.core import Request, SamplingParams, StreamChunk
from cs336_basics.mini_sglang.engine import ModelEngine
from cs336_basics.mini_sglang.kv_cache import KVPool, RadixPrefixCache
from cs336_basics.mini_sglang.scheduler import Scheduler
from cs336_basics.mini_sglang.server import ServerState, create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger("mini_sglang")


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args) -> tuple:
    """Load the TransformerLM + GPT-2 tokenizer from a checkpoint."""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    device = _resolve_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("model_config", {})

    model_kwargs = {
        "vocab_size":     cfg.get("vocab_size",     args.vocab_size),
        "context_length": cfg.get("context_length", args.context_length),
        "num_layers":     cfg.get("num_layers",     args.num_layers),
        "d_model":        cfg.get("d_model",        args.d_model),
        "num_heads":      cfg.get("num_heads",      args.num_heads),
        "d_ff":           cfg.get("d_ff",           args.d_ff),
        "rope_theta":     cfg.get("rope_theta",     args.rope_theta),
    }

    model = TransformerLM(**model_kwargs).to(device)
    sd = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
    model.load_state_dict(sd)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded model: %s params on %s", f"{n_params:,}", device)
    logger.info("Model config: %s", model_kwargs)

    return model, tokenizer, model_kwargs, device


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_scheduler(model, tokenizer, model_kwargs, device, args):
    engine = ModelEngine(model, device)

    kv_pool = KVPool(
        num_layers=model_kwargs["num_layers"],
        num_heads=model_kwargs["num_heads"],
        head_dim=model_kwargs["d_model"] // model_kwargs["num_heads"],
        pool_size=args.kv_pool_size,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    prefix_cache = RadixPrefixCache(kv_pool)

    scheduler = Scheduler(
        engine=engine,
        kv_pool=kv_pool,
        prefix_cache=prefix_cache,
        max_running=args.max_running,
        max_context_len=model_kwargs["context_length"],
    )

    return scheduler


# ═══════════════════════════════════════════════════════════════════════
# Interactive CLI mode
# ═══════════════════════════════════════════════════════════════════════

async def _interactive(scheduler: Scheduler, tokenizer, args):
    """Simple REPL: type a prompt, see generated text."""
    task = asyncio.create_task(scheduler.run())

    uid = 0
    print("\n╭─ mini-sglang interactive mode ─────────────────────────────╮")
    print("│  Type a prompt and press Enter.  Type 'quit' to exit.     │")
    print("│  Type 'stats' to see scheduler statistics.                │")
    print("╰───────────────────────────────────────────────────────────╯\n")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_ids = [eos_id] if eos_id is not None else []

    try:
        while True:
            try:
                prompt = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("prompt> ")
                )
            except EOFError:
                break

            prompt = prompt.strip()
            if not prompt:
                continue
            if prompt.lower() == "quit":
                break
            if prompt.lower() == "stats":
                s = scheduler.get_stats()
                print(f"  pending={s.pending_requests}  running={s.running_requests}  "
                      f"total={s.total_requests}  prefill_tok={s.total_prefill_tokens}  "
                      f"decode_tok={s.total_decode_tokens}  cache_hits={s.total_cache_hit_tokens}  "
                      f"pool_free={s.kv_pool_free}/{s.kv_pool_total}  "
                      f"prefix_cached={s.prefix_cache_entries}")
                continue

            uid += 1
            input_ids = tokenizer.encode(prompt)
            output_queue: asyncio.Queue[StreamChunk] = asyncio.Queue()

            req = Request(
                uid=uid,
                input_ids=input_ids,
                sampling_params=SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_tokens,
                    stop_token_ids=stop_ids,
                ),
                output_queue=output_queue,
            )
            await scheduler.add_request(req)

            # Stream tokens to stdout
            tokens = []
            prev_text = ""
            sys.stdout.write("\n")
            while True:
                chunk: StreamChunk = await output_queue.get()
                if chunk.token_id >= 0:
                    tokens.append(chunk.token_id)
                full = tokenizer.decode(tokens)
                delta = full[len(prev_text):]
                prev_text = full
                sys.stdout.write(delta)
                sys.stdout.flush()
                if chunk.finished:
                    break

            ttft = (req.first_token_time - req.arrival_time) * 1000 if req.first_token_time else 0
            total = (time.time() - req.arrival_time) * 1000
            print(f"\n  [{len(tokens)} tokens, TTFT {ttft:.1f}ms, total {total:.1f}ms]\n")

    finally:
        scheduler.stop()
        task.cancel()


# ═══════════════════════════════════════════════════════════════════════
# Throughput benchmark
# ═══════════════════════════════════════════════════════════════════════

async def _benchmark(scheduler: Scheduler, tokenizer, args):
    """Fire N concurrent requests and report throughput."""
    task = asyncio.create_task(scheduler.run())

    prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far far away",
        "To be or not to be",
    ]
    n = args.bench_n
    max_tok = args.bench_max_tokens

    eos_id = getattr(tokenizer, "eos_token_id", None)
    stop_ids = [eos_id] if eos_id is not None else []

    print(f"\n  Benchmark: {n} requests, max_new_tokens={max_tok}")

    reqs = []
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        oq: asyncio.Queue[StreamChunk] = asyncio.Queue()
        req = Request(
            uid=i + 1,
            input_ids=tokenizer.encode(prompt),
            sampling_params=SamplingParams(
                max_new_tokens=max_tok,
                stop_token_ids=stop_ids,
            ),
            output_queue=oq,
        )
        reqs.append(req)

    t0 = time.time()
    for req in reqs:
        await scheduler.add_request(req)

    total_tokens = 0
    for req in reqs:
        while True:
            chunk = await req.output_queue.get()
            if chunk.token_id >= 0:
                total_tokens += 1
            if chunk.finished:
                break

    elapsed = time.time() - t0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    print(f"  Completed {n} requests in {elapsed:.2f}s")
    print(f"  Total generated tokens: {total_tokens}")
    print(f"  Throughput: {throughput:.1f} tokens/s")

    s = scheduler.get_stats()
    print(f"  Cache hits: {s.total_cache_hit_tokens} tokens")
    print(f"  Prefix cache entries: {s.prefix_cache_entries}\n")

    scheduler.stop()
    task.cancel()


# ═══════════════════════════════════════════════════════════════════════
# HTTP server mode
# ═══════════════════════════════════════════════════════════════════════

def _run_server(scheduler: Scheduler, state: ServerState, args):
    import uvicorn

    app = create_app(state)

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(scheduler.run())
        logger.info("Scheduler background task started")

    @app.on_event("shutdown")
    async def _shutdown():
        scheduler.stop()

    logger.info("Starting HTTP server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="mini-sglang: lightweight LLM serving engine"
    )

    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to TransformerLM checkpoint (.pt)")
    p.add_argument("--device", type=str, default="auto")

    # Model config fallbacks
    g = p.add_argument_group("model (fallback if not in checkpoint)")
    g.add_argument("--vocab_size",     type=int,   default=50257)
    g.add_argument("--context_length", type=int,   default=256)
    g.add_argument("--num_layers",     type=int,   default=4)
    g.add_argument("--d_model",        type=int,   default=512)
    g.add_argument("--num_heads",      type=int,   default=16)
    g.add_argument("--d_ff",           type=int,   default=1344)
    g.add_argument("--rope_theta",     type=float, default=10000.0)

    # Serving config
    g = p.add_argument_group("serving")
    g.add_argument("--host",         type=str, default="0.0.0.0")
    g.add_argument("--port",         type=int, default=8000)
    g.add_argument("--kv_pool_size", type=int, default=65536,
                    help="Number of KV cache slots to pre-allocate")
    g.add_argument("--max_running",  type=int, default=64,
                    help="Max concurrent decoding requests")

    # Mode selection
    g = p.add_argument_group("mode")
    g.add_argument("--interactive", action="store_true",
                    help="Interactive CLI mode instead of HTTP server")
    g.add_argument("--benchmark",   action="store_true",
                    help="Run a quick throughput benchmark")
    g.add_argument("--bench_n",     type=int, default=8,
                    help="Number of requests for benchmark")
    g.add_argument("--bench_max_tokens", type=int, default=32)

    # Generation defaults (interactive mode)
    g = p.add_argument_group("generation defaults (interactive)")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top_p",      type=float, default=0.9)
    g.add_argument("--max_tokens", type=int,   default=128)

    return p


def main():
    args = build_parser().parse_args()

    model, tokenizer, model_kwargs, device = load_model(args)
    scheduler = _build_scheduler(model, tokenizer, model_kwargs, device, args)

    if args.interactive:
        asyncio.run(_interactive(scheduler, tokenizer, args))
    elif args.benchmark:
        asyncio.run(_benchmark(scheduler, tokenizer, args))
    else:
        state = ServerState(scheduler, tokenizer)
        _run_server(scheduler, state, args)


if __name__ == "__main__":
    main()
