# Running CS336 Assignment on GPU K8s Cluster

## Overview

This guide shows you how to run the CS336 assignment on your GPU-enabled Kubernetes cluster.

## Complexity Level: **Low to Medium**

The good news: PyTorch makes GPU usage very simple! You only need to:
1. Build a Docker image with CUDA support
2. Deploy to K8s with GPU resources
3. Add a few lines to your Python code

---

## Step 1: Prepare Your Code for GPU

### Minimal Code Changes Needed

In your training script, add these lines:

```python
import torch

# Detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move model to GPU
model = YourTransformer(...).to(device)

# Move data to GPU during training
for batch in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Training code...
```

That's it! PyTorch handles everything else automatically.

---

## Step 2: Build Docker Image

### Using the provided Dockerfile:

```bash
# Build the image
docker build -t your-registry/cs336-assignment:latest .

# Test locally (if you have GPU)
docker run --gpus all -it your-registry/cs336-assignment:latest

# Push to your registry
docker push your-registry/cs336-assignment:latest
```

### What the Dockerfile includes:
- NVIDIA CUDA 12.1 base image
- Python 3.10
- UV package manager (fast!)
- All your project dependencies

---

## Step 3: Deploy to K8s

### Option A: Simple Pod (for testing)

```bash
# Edit k8s-gpu-pod.yaml to update image registry
# Then deploy:
kubectl apply -f k8s-gpu-pod.yaml

# Check status
kubectl get pods
kubectl logs cs336-training -f

# Get a shell
kubectl exec -it cs336-training -- /bin/bash
```

### Option B: Job (for training runs)

Create `k8s-training-job.yaml`:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: cs336-training-job
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: your-registry/cs336-assignment:latest
        command: ["uv", "run", "python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-h100  # or your GPU type
```

---

## Step 4: Verify GPU Access

Once your pod is running:

```bash
kubectl exec -it cs336-training -- python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

---

## Development Workflow

### Recommended approach:

1. **Develop locally** (on your current machine)
   - Implement BPE tokenizer (CPU)
   - Implement Transformer model (CPU)
   - Run unit tests: `uv run pytest`
   - Test with small data on CPU

2. **Test in K8s with GPU**
   - Build Docker image
   - Deploy to K8s
   - Run small training test (few iterations)
   - Verify GPU is being used

3. **Full training run**
   - Submit K8s Job
   - Monitor with `kubectl logs -f`
   - Save checkpoints to persistent volume

---

## Tips for K8s GPU Training

### 1. Use Persistent Volumes for Data
```yaml
volumes:
- name: data
  persistentVolumeClaim:
    claimName: cs336-data-pvc
```

### 2. Save Checkpoints Regularly
```python
# In your training code
if iteration % 1000 == 0:
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }, f'/workspace/checkpoints/checkpoint_{iteration}.pt')
```

### 3. Monitor GPU Usage
```bash
# From inside the pod
nvidia-smi

# Or watch it
watch -n 1 nvidia-smi
```

### 4. Handle Interruptions
```python
import signal
import sys

def save_checkpoint_and_exit(signum, frame):
    print("Saving checkpoint before exit...")
    save_checkpoint(model, optimizer, iteration)
    sys.exit(0)

signal.signal(signal.SIGTERM, save_checkpoint_and_exit)
```

---

## Estimated Resource Needs

Based on the assignment:

### For TinyStories Training:
- **GPU**: 1x H100 or A100
- **Memory**: 16-32 GB
- **Time**: ~5-10 hours total for all experiments

### For OpenWebText Training:
- **GPU**: 1x H100
- **Memory**: 32-64 GB
- **Time**: ~3 hours for main experiment

---

## Common Issues and Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in your training script

### Issue: "No GPU detected"
**Solution**: Check node selector and GPU resource requests

### Issue: "Pod stuck in Pending"
**Solution**: Check if GPU nodes are available:
```bash
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```

---

## Quick Start Commands

```bash
# 1. Build and push image
docker build -t gcr.io/your-project/cs336:latest .
docker push gcr.io/your-project/cs336:latest

# 2. Create data PVC (if needed)
kubectl apply -f k8s-pvc.yaml

# 3. Run training
kubectl apply -f k8s-gpu-pod.yaml

# 4. Monitor
kubectl logs -f cs336-training

# 5. Access pod
kubectl exec -it cs336-training -- /bin/bash
```

---

## Summary

**Complexity**: ⭐⭐☆☆☆ (2/5)

- Docker setup: Simple (provided Dockerfile)
- K8s deployment: Medium (provided YAML)
- Code changes: Minimal (just `.to(device)`)
- GPU usage: Automatic (PyTorch handles it)

**You can do this!** The infrastructure setup is straightforward, and PyTorch makes GPU usage almost transparent.
