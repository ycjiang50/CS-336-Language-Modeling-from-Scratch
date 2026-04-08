import torch
import torch.distributed as dist
import time
import os
import argparse
from torch.multiprocessing.spawn import spawn

def setup(master_addr, master_port, rank, world_size, backend):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = master_addr  # Master address; other ranks connect here
    os.environ['MASTER_PORT'] = str(master_port)
    # Initialize the process group for this backend
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # Common backends: gloo (CPU), nccl (GPU), mpi, etc.

def cleanup():
    """Tear down the process group."""
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_all_reduce(rank, world_size, tensor_size_mb, backend, device, master_addr, master_port):
    """Run all-reduce timing and bandwidth measurement."""
    # 1. Environment and device
    setup(master_addr, master_port, rank, world_size, backend)
    if device == 'cuda':
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

    # 2. Allocate test tensor
    tensor_size_bytes = tensor_size_mb * 1024 * 1024
    # float32 = 4 bytes per element
    num_elements = tensor_size_bytes // 4
    tensor_data = torch.randn(num_elements, device=device)

    # 3. Warm-up (important for stable timings)
    for _ in range(5):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)
        if device == 'cuda':
            torch.cuda.synchronize()

    dist.barrier()
    # 4. Timed runs
    start_time = time.time()
    num_iterations = 20
    for _ in range(num_iterations):
        dist.all_reduce(tensor_data, op=dist.ReduceOp.SUM)

    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    duration = end_time - start_time
    avg_time = duration / num_iterations

    # Bandwidth (every rank computes so return values are consistent)
    # Throughput ≈ tensor_size / time
    bandwidth_gbps = (tensor_size_bytes / avg_time) / 1e9

    if rank == 0:
        print(f"Backend: {backend}, Device: {device}, World Size: {world_size}, Tensor Size: {tensor_size_mb}MB")
        print(f"Average time per all-reduce: {avg_time * 1000:.4f} ms")
        print(f"Achieved Bandwidth: {bandwidth_gbps:.4f} GB/s\n")

    local_result = {
            'rank': rank,
            'world_size': world_size,
            'backend': backend,
            'device': device,
            'tensor_size_mb': tensor_size_mb,
            'avg_time_ms': avg_time * 1000.0,
            'bandwidth_gbps': bandwidth_gbps
        }
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_result)

    cleanup()
    if rank == 0:
        return gathered_results
