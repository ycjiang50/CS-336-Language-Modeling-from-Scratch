from __future__ import annotations

import timeit
import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml
from pathlib import Path
from cs336_basics.model import BasicsTransformerLM # type: ignore
from cs336_basics.optimizer import AdamW, get_cosine_lr
import os
from contextlib import nullcontext

try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
    print("=" * 60)
    print("✓ NVTX IS AVAILABLE in benchmarking script")
    print("=" * 60)
except (ImportError, AttributeError) as e:
    # If NVTX is not available, create a dummy context manager
    NVTX_AVAILABLE = False
    print("=" * 60)
    print(f"✗ WARNING: NVTX NOT AVAILABLE (error: {e})")
    print("  Profiling annotations will NOT appear in the profile!")
    print("=" * 60)
    class DummyNVTX:
        @staticmethod
        def range(name):
            class DummyContextManager:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyContextManager()
    nvtx = DummyNVTX()

print (os.getcwd())

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str) -> torch.Tensor:
    """Create a random batch of token IDs for benchmarking."""
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device)


def benchmark_model(
    model: nn.Module,
    batch: torch.Tensor,
    warmup_steps: int,
    benchmark_steps: int,
    forward_only: bool,
    device: str,
    autocast: bool
):
    """
    Benchmark the model's forward and backward passes.
    
    Args:
        model: The model to benchmark
        batch: Input batch tensor
        warmup_steps: Number of warmup steps
        benchmark_steps: Number of steps to benchmark
        forward_only: Whether to only benchmark forward pass
        device: Device to run on
        
    Returns:
        Tuple of (forward_time, backward_time) in seconds
    """
    print ("warmup")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast else nullcontext()
    # Warmup - use NVTX to mark this so you can filter it out in the profiler
    with nvtx.range("WARMUP"):
        with ctx: 
            for step in range(warmup_steps):
                with nvtx.range(f"warmup_step_{step}"):
                    lr = 0.1
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    
                    with nvtx.range("warmup_forward"):
                        outputs = model(batch)
                    
                    if not forward_only:
                        with nvtx.range("warmup_backward"):
                            with nvtx.range("warmup_zero_grad"):
                                optimizer.zero_grad()
                            with nvtx.range("warmup_loss_backward"):
                                loss = outputs.mean()
                                loss.backward()
                            with nvtx.range("warmup_optimizer_step"):
                                optimizer.step()
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
    
    # Start recording memory history AFTER warm-up
    if device == "cuda":
        print("Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Benchmark - use NVTX to mark the actual benchmarking region
    print ("benchmark:")
    forward_times = []
    backward_times = []
    with nvtx.range("BENCHMARK"):
        with ctx: 
            for step in range(benchmark_steps):
                with nvtx.range(f"benchmark_step_{step}"):
                    lr = 0.1
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    
                    # Forward pass - clearly marked for profiling
                    with nvtx.range("FORWARD"):
                        start_time = timeit.default_timer()
                        outputs = model(batch)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        forward_time = timeit.default_timer() - start_time
                        forward_times.append(forward_time)
                    
                    if not forward_only:
                        # Backward pass - split into gradient computation and optimizer
                        start_time = timeit.default_timer()
                        
                        with nvtx.range("ZERO_GRAD"):
                            optimizer.zero_grad()
                        
                        with nvtx.range("BACKWARD"):
                            loss = outputs.mean()
                            loss.backward()
                        
                        with nvtx.range("OPTIMIZER_STEP"):
                            optimizer.step()
                        
                        if device == "cuda":
                            torch.cuda.synchronize()
                        backward_time = timeit.default_timer() - start_time
                        backward_times.append(backward_time)

        # Memory measurement - separate NVTX region
        with nvtx.range("MEMORY_MEASUREMENT"):
            torch.cuda.reset_peak_memory_stats()
            with nvtx.range("memory_forward"):
                outputs = model(batch)
                torch.cuda.synchronize()
            memory_before_backward = torch.cuda.max_memory_allocated()/(1024**3)

            memory_backward=0
            if not forward_only:
                torch.cuda.reset_peak_memory_stats()
                with nvtx.range("memory_backward"):
                    with nvtx.range("memory_zero_grad"):
                        optimizer.zero_grad()
                    with nvtx.range("memory_loss_backward"):
                        loss = outputs.mean()
                        loss.backward()
                    with nvtx.range("memory_optimizer_step"):
                        optimizer.step()
                    torch.cuda.synchronize()
                memory_backward = torch.cuda.max_memory_allocated()/(1024**3)
    
    # Save memory snapshot and stop recording
    if device == "cuda":
        print("Saving memory snapshot...")
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Memory snapshot saved to: memory_snapshot.pickle")
    
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times) if not forward_only else 0.0
    
    return avg_forward_time, avg_backward_time, memory_before_backward, memory_backward




def main():
    # Load configuration
    config_path = Path("configures/xl.yaml")
    config = load_config(config_path)
    
    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    ).to(config["device"])

    #model = torch.compile(model)
    
    # Create random batch
    batch = create_random_batch(
        config["batch_size"],
        config["context_length"],
        config["vocab_size"],
        config["device"]
    )
    
    # Run benchmarkl save
    for autocast in [True]:
        forward_time, backward_time, forward_memory, backward_memory = benchmark_model(
            model,
            batch,
            config["warmup_steps"],
            config["benchmark_steps"],
            config["forward_only"],
            config["device"],
            autocast
        )
        
        # Print results
        print(f"\nBenchmark Results:")
        print(f"Model Configuration:")
        print(f"autocast to bf16: {autocast}")
        # print(f"  - Layers: {config['num_layers']}")
        # print(f"  - Model Dimension: {config['d_model']}")
        # print(f"  - Heads: {config['num_heads']}")
        # print(f"  - FF Dimension: {config['d_ff']}")
        # print(f"  - Context Length: {config['context_length']}")
        # print(f"  - Batch Size: {config['batch_size']}")
        print(f"\nTiming Results:")
        print(f"  - Average Forward Time: {forward_time*1000:.2f} ms")
        print(f"forward peak memory: {forward_memory}")
        print(f"backward memory:{backward_memory}")
        if not config["forward_only"]:
            print(f"  - Average Backward Time: {backward_time*1000:.2f} ms")
            print(f"  - Total Time: {(forward_time + backward_time)*1000:.2f} ms")


if __name__ == "__main__":
    main() 