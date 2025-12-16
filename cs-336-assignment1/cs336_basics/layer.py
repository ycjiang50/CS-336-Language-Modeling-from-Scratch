import torch
import math
from typing import Optional, Union, List, Tuple
from torch import nn
from torch.nn import init
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (in_features + out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x):
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings, # vocabulary size
                 embedding_dim, # embedding dimension
                 device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Weight shape is (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        # Initialize weights using truncated normal
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Initialize weights to 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Prevent overflow in mean/sqrt calculations
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Perform RMSNorm calculation 
        
        # official implementation:
        # rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) 
        # normalized_x = x * rms

        # our implementation:
        RMS = (x.pow(2).mean(dim=-1, keepdim=True)+ self.eps).sqrt() 
        normalized_x = x / RMS
        
        results = normalized_x * self.weight # W will automatically broadcast to ..., d_model

        # Return the result in the original dtype
        return results.to(in_dtype)

def silu(x: torch.Tensor): return x * torch.sigmoid(x) # SiLU activation function

def glu(a:torch.Tensor, b:torch.Tensor): return a * b # element-wise multiplication

def swiglu_fn(a: torch.Tensor, b: torch.Tensor): return glu(silu(a), b) # SwiGLU activation function

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.linear1 = Linear(d_model, d_ff, **factory_kwargs)  # W1
        self.linear2 = Linear(d_ff, d_model, **factory_kwargs)  # W2
        self.linear3 = Linear(d_model, d_ff, **factory_kwargs)  # W3

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.linear1(x)
        w3x = self.linear3(x)
        h = swiglu_fn(w1x, w3x)
        return self.linear2(h)

def get_compatible_dff(d_model: int) -> int:
    """
    Returns the nearest multiple of 64 to 8/3 * d_model.
    """
    raw = (8 * d_model) / 3
    rounded = int((raw + 32) // 64) * 64  # round to nearest multiple of 64
    return rounded

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) layer.

    Args
    ----
    theta : float
        Base used to generate inverse frequencies (e.g. 10_000).
    d_k : int
        Dimension of the key / query vectors (must be even).
    max_seq_len : int
        Maximum sequence length expected at inference / training time.
    device : torch.device | None
        Where to place the pre-computed sine / cosine tables.
    """
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        self.d_k = d_k
        # ---- pre-compute inverse frequencies ----
        # freq[k] = 1 / theta ** (2k / d_k)          (k = 0,1,…,d_k/2-1)
        freq = 1.0 / (theta ** (torch.arange(0,d_k,2, device=device).float() / d_k))

        # shape: (max_seq_len, d_k // 2)
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        # cache cos/sin; no gradients needed → persistent=False
        self.register_buffer('cos_cached', torch.cos(freqs),persistent=False) # persistent=False does not save to state_dict
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"]
        ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """
        Apply RoPE to `x`.  Works with any batch shape prefix.
        """
        # Check if the last dimension matches d_k
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        # Gather the cached tables for the required positions
        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]

        # If x is 4D (B, H, S, D) and we have cos_pos (B, S, D), we need to unsqueeze for H
        if x.ndim == 4 and cos_pos.ndim == 3:
            cos_pos = cos_pos.unsqueeze(1)
            sin_pos = sin_pos.unsqueeze(1)

        # Split even / odd channels
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # Apply the 2-D rotation to each pair
        out_even = x_even * cos_pos - x_odd * sin_pos
        out_odd = x_even * sin_pos + x_odd * cos_pos

        # Re-interleave
        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(
        self,
        query: Float[torch.Tensor, "... seq_len_q d_k"],
        key: Float[torch.Tensor, "... seq_len_k d_k"],
        value: Float[torch.Tensor, "... seq_len_k d_v"],
        mask: Bool[torch.Tensor, "seq_len_q seq_len_k"] = None
    ) -> Float[torch.Tensor, "... seq_len_q d_v"]:
        # Compute scaled dot product attention scores using einsum
        attn_scores = einsum(query, key, "... q d, ... k d -> ... q k") * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = softmax_stable(attn_scores, dim=-1)

        # Compute attention output using einsum again
        output = einsum(attn_probs, value, "... q k, ... k d -> ... q d")

        return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k  # match d_k for simplicity
        self.use_rope = use_rope

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **factory_kwargs)
                                                              for _ in range(4)]
        self.attn = ScaledDotProductAttention(self.d_k)

        # Create a causal mask for the attention mechanism
        # Shape: (1, 1, max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
        token_positions: Int[torch.Tensor, "batch seq_len"]| None = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[Float[torch.Tensor, "batch seq_len d_model"], tuple[Float[torch.Tensor, "batch seq_len d_model"], tuple[torch.Tensor, torch.Tensor]]]:
        B, S, _ = x.shape

        # Project to multi-head Q, K, V
        q,k,v = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads) 
                 for proj in [self.q_proj, self.k_proj, self.v_proj]]

        # Apply RoPE to Q and K if enabled
        if self.use_rope: q,k = self.rope(q, token_positions),self.rope(k, token_positions)

        # KV Cache handling
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        current_key_value = (k, v) if use_cache else None

        # Compute attention
        # If using cache, we are generating 1 token, but attending to S_past + 1 tokens.
        # The mask needs to handle this.
        # If past_key_value is None, we are in the first step (prefill), S is full sequence length.
        # If past_key_value is not None, S is 1 (usually), and total sequence length is S_past + S.
        
        total_seq_len = k.shape[2]
        # We only need to mask if we are in prefill (query length > 1) or if we want to be safe.
        # For generation step (query length 1), we attend to all past tokens and current token.
        # The causal mask slice should correspond to the query positions.
        
        if past_key_value is not None:
             # Query is at the end of the sequence
             # We want to attend to all previous keys.
             # The mask for a single query token at pos P attending to keys 0..P is all ones (if keys are 0..P).
             # Our causal_mask is (1, 1, max_len, max_len).
             # We need mask[..., start_pos:start_pos+S, :total_seq_len]
             start_pos = total_seq_len - S
             mask = self.causal_mask[..., start_pos:start_pos+S, :total_seq_len]
        else:
             mask = self.causal_mask[..., :S, :S]

        out = self.attn(q, k, v, mask=mask)

        # Merge heads and project
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.o_proj(out)
        
        if use_cache:
            return out, current_key_value
        return out


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with two sub-layers:

       x ──► RMSNorm ──► MHA ──► + ──►
         │                     ▲
         └─────────────────────┘     (sublayer-1)

       y ──► RMSNorm ──► FF  ──► + ──► out
         │                     ▲
         └─────────────────────┘     (sublayer-2)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}

        # ── sub-layer 1: (RMSNorm → causal MHA) ──────────────────────────────
        self.norm1 = RMSNorm(d_model, **kwargs)
        self.attn  = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=use_rope,
            **kwargs,
        )

        # ── sub-layer 2: (RMSNorm → feed-forward) ────────────────────────────
        self.norm2 = RMSNorm(d_model, **kwargs)
        self.ff    = SwiGLU(d_model=d_model, d_ff=d_ff, **kwargs)

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # (batch, seq_len)
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        b, s, _ = x.shape

        # ---- sub-layer-1: RMSNorm → MHA → residual -------------------------
        if use_cache:
            attn_out, current_key_value = self.attn(self.norm1(x), token_positions=token_positions, past_key_value=past_key_value, use_cache=True)
        else:
            attn_out = self.attn(self.norm1(x), token_positions=token_positions, past_key_value=past_key_value, use_cache=False)
            current_key_value = None
            
        x = x + attn_out                       # residual connection

        # ---- sub-layer-2: RMSNorm → FF → residual --------------------------
        ff_out   = self.ff(self.norm2(x))
        x        = x + ff_out                  # residual connection
        
        if use_cache:
            return x, current_key_value
        return x

def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """
    Copy `source` into `target` in-place, transposing `source` if that
    is what makes the shapes line up.
    """
    if source.shape == target.shape:
        target.data.copy_(source)
    elif source.T.shape == target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"Shape mismatch: cannot load parameter of shape {source.shape} "
                         f"into tensor of shape {target.shape}")
    

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)

        # token embedding  (no separate pos-emb: RoPE lives inside blocks)
        self.tok_emb = Embedding(vocab_size, d_model, **kw)

        # L Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                use_rope=True,
                **kw,
            )
            for _ in range(num_layers)
        ])

        # final norm
        self.ln_final = RMSNorm(d_model, **kw)
        self.lm_head = Linear(d_model, vocab_size, **kw)

        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor, past_key_values: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, List[tuple[torch.Tensor, torch.Tensor]]]]:
        b, s = token_ids.shape
        
        # Calculate start position based on cache
        if past_key_values is not None:
            # past_key_values[0][0] is (batch, heads, seq_len_past, d_head)
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 0
            
        if s + past_len > self.context_length:
            raise ValueError(f"seq_len {s} + past_len {past_len} exceeds context_length {self.context_length}")

        # token embeddings
        x = self.tok_emb(token_ids)                         # (b, s, d)

        # token positions for RoPE
        # positions should be from past_len to past_len + s
        pos = torch.arange(past_len, past_len + s, device=token_ids.device).unsqueeze(0).expand(b, s)

        # transformer stack
        new_past_key_values = [] if use_cache else None
        
        for i, blk in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if use_cache:
                x, new_kv = blk(x, token_positions=pos, past_key_value=past_kv, use_cache=True)
                new_past_key_values.append(new_kv)
            else:
                x = blk(x, token_positions=pos, past_key_value=past_kv, use_cache=False)

        # final norm → tied linear projection (logits)
        x = self.ln_final(x)                                 # (b, s, d)

        logits = self.lm_head(x)  # (b, s, vocab_size)
        
        if use_cache:
            return logits, new_past_key_values
        return logits