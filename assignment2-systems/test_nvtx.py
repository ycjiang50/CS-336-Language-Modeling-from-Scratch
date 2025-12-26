#!/usr/bin/env python
"""Quick test to verify NVTX is working."""

print("Testing NVTX availability...")
print("=" * 60)

# Test 1: Can we import torch.cuda.nvtx?
try:
    import torch.cuda.nvtx as nvtx
    print("✓ torch.cuda.nvtx import: SUCCESS")
except Exception as e:
    print(f"✗ torch.cuda.nvtx import: FAILED")
    print(f"  Error: {e}")
    exit(1)

# Test 2: Can we use nvtx.range?
try:
    with nvtx.range("test_range"):
        pass
    print("✓ nvtx.range() context manager: SUCCESS")
except Exception as e:
    print(f"✗ nvtx.range() failed")
    print(f"  Error: {e}")
    exit(1)

# Test 3: Check if model.py will work
try:
    from cs336_basics.model import BasicsTransformerLM
    print("✓ cs336_basics.model import: SUCCESS")
    print("  Check above for NVTX availability message from model.py")
except Exception as e:
    print(f"✗ cs336_basics.model import: FAILED")
    print(f"  Error: {e}")
    exit(1)

print("=" * 60)
print("All tests passed! NVTX annotations should work.")
print("=" * 60)
