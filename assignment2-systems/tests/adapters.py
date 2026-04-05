from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# FlashAttention – Pure PyTorch (tiled online softmax, no Triton)
# ═══════════════════════════════════════════════════════════════════════

class _FlashAttentionPyTorch(torch.autograd.Function):
    """FlashAttention-2 implemented with standard PyTorch ops."""

    _TILE = 64

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        B, Nq, d = q.shape
        Nk = k.shape[1]
        scale = d ** -0.5
        T = _FlashAttentionPyTorch._TILE

        O = torch.zeros_like(q)
        L = torch.zeros(B, Nq, device=q.device, dtype=q.dtype)

        for b in range(B):
            for i in range(0, Nq, T):
                ie = min(i + T, Nq)
                qi = q[b, i:ie]
                oi = torch.zeros_like(qi)
                li = torch.zeros(qi.shape[0], 1, device=q.device, dtype=torch.float32)
                mi = torch.full((qi.shape[0], 1), float("-inf"), device=q.device, dtype=torch.float32)

                for j in range(0, Nk, T):
                    je = min(j + T, Nk)
                    kj, vj = k[b, j:je], v[b, j:je]
                    sij = (qi @ kj.T * scale).float()

                    if is_causal:
                        qp = torch.arange(i, ie, device=q.device).unsqueeze(1)
                        kp = torch.arange(j, je, device=q.device).unsqueeze(0)
                        sij = sij.masked_fill(qp < kp, -1e6)

                    mij = sij.max(dim=-1, keepdim=True).values
                    mi_new = torch.maximum(mi, mij)
                    pij = torch.exp(sij - mij)

                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(mij - mi_new)
                    li = alpha * li + beta * pij.sum(dim=-1, keepdim=True)
                    oi = alpha * oi + beta * (pij.to(vj.dtype) @ vj)
                    mi = mi_new

                O[b, i:ie] = oi / li
                L[b, i:ie] = (mi + torch.log(li)).squeeze(-1)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        B, Nq, d = q.shape
        Nk = k.shape[1]
        scale = d ** -0.5
        T = _FlashAttentionPyTorch._TILE

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        D = (dO * O).sum(dim=-1, keepdim=True)  # (B, Nq, 1)

        for b in range(B):
            for j in range(0, Nk, T):
                je = min(j + T, Nk)
                kj, vj = k[b, j:je], v[b, j:je]
                dkj = torch.zeros_like(kj)
                dvj = torch.zeros_like(vj)

                for i in range(0, Nq, T):
                    ie = min(i + T, Nq)
                    qi = q[b, i:ie]
                    doi = dO[b, i:ie]
                    li = L[b, i:ie].unsqueeze(-1)
                    di = D[b, i:ie]

                    sij = (qi @ kj.T * scale).float()
                    if is_causal:
                        qp = torch.arange(i, ie, device=q.device).unsqueeze(1)
                        kp = torch.arange(j, je, device=q.device).unsqueeze(0)
                        sij = sij.masked_fill(qp < kp, -1e6)

                    pij = torch.exp(sij - li)
                    dvj += pij.to(doi.dtype).T @ doi
                    dpij = doi @ vj.T
                    dsij = (pij * (dpij.float() - di) * scale).to(qi.dtype)
                    dq[b, i:ie] += dsij @ kj
                    dkj += dsij.T @ qi

                dk[b, j:je] += dkj
                dv[b, j:je] += dvj

        return dq, dk, dv, None


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    return _FlashAttentionPyTorch


# ═══════════════════════════════════════════════════════════════════════
# FlashAttention – Triton
# ═══════════════════════════════════════════════════════════════════════

def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    from cs336_systems.flash_attention_backward import FlashAttentionAutogradFunctionTriton
    return FlashAttentionAutogradFunctionTriton


# ═══════════════════════════════════════════════════════════════════════
# DDP – Individual Parameters
# ═══════════════════════════════════════════════════════════════════════

class _DDPIndividualParameters(nn.Module):
    """DDP wrapper that syncs each parameter's gradient individually via
    async allreduce hooks registered during __init__."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self._handles: list = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        seen: set = set()
        for param in self.module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                param.register_hook(self._make_hook())

    def _make_hook(self):
        def hook(grad):
            handle = dist.all_reduce(grad, async_op=True)
            self._handles.append(handle)
        return hook

    def forward(self, *args, **kwargs):
        self._handles.clear()
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self._handles:
            handle.wait()
        seen: set = set()
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None and id(param) not in seen:
                seen.add(id(param))
                param.grad.div_(self.world_size)
        self._handles.clear()


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    return _DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()


# ═══════════════════════════════════════════════════════════════════════
# DDP – Bucketed
# ═══════════════════════════════════════════════════════════════════════

def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    from cs336_systems.ddp_overlap_bucked import DDPOverlapBucketed
    return DDPOverlapBucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    ddp_model.finish_gradient_synchronization()


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # Bucket state reset is handled inside DDPOverlapBucketed.forward()
    pass


# ═══════════════════════════════════════════════════════════════════════
# Sharded Optimizer (ZeRO Stage-1)
# ═══════════════════════════════════════════════════════════════════════

class _ShardedOptimizer:
    """Each rank maintains optimizer state for only its shard of parameters.
    After each step the updated parameters are broadcast to all ranks."""

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
        self.all_params = list(params)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self._owner = {id(p): i % self.world_size for i, p in enumerate(self.all_params)}

        owned = [p for p in self.all_params if self._owner[id(p)] == self.rank]
        self._local_opt = optimizer_cls(owned, **kwargs) if owned else None

    def zero_grad(self, set_to_none: bool = True):
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        if self._local_opt is not None:
            self._local_opt.step()
        for p in self.all_params:
            dist.broadcast(p.data, src=self._owner[id(p)])


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return _ShardedOptimizer(params, optimizer_cls, **kwargs)
