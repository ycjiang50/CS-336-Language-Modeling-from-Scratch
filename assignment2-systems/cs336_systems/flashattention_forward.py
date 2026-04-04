import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # program_id(0) = Q tile index, program_id(1) = batch index
    i = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer by batch_index * batch_stride
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),  # K, V start at offset 0; advanced in the inner loop
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(i * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, 1),
        order=(1, 0),
    )

    # Causal masking: the goal is to zero out (set to -inf) the upper-triangle
    # region of the full S matrix.  Because we compute S in tiles S_ij:
    #   - If tile i < j: the entire tile lies in the upper triangle → mask all.
    #   - If tile i > j: the entire tile lies in the lower triangle → keep all.
    #   - If tile i == j: the tile straddles the diagonal → per-element mask.
    # We handle all three cases with a single positional mask below.

    Q_i = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)
    O_i_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i_acc = tl.zeros((Q_TILE_SIZE, 1), dtype=tl.float32)
    M_i_acc = tl.full((Q_TILE_SIZE, 1), float('-inf'), dtype=tl.float32)

    # Inner loop: iterate over K/V tiles along the key dimension
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
        V_j = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

        # Scaled dot-product attention scores
        S_ij = tl.dot(Q_i, K_j.T) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal mask using global position indices so that a single
        # expression covers all tile positions (Triton compiles away the
        # branch on the constexpr is_causal flag).
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]  # (Q_TILE_SIZE, 1)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)[None, :]  # (1, K_TILE_SIZE)
            causal_mask = q_idx >= k_idx  # True for lower triangle (positions to keep)
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # Online softmax: use local block max for numerical stability,
        # then apply a correction factor when merging with the running state.
        M_ij = tl.max(S_ij, axis=1, keep_dims=True)       # local max of current block
        M_i_new = tl.maximum(M_i_acc, M_ij)                # updated running max
        P_ij = tl.exp(S_ij - M_ij)                         # exp with local max (values in [0, 1])

        # Merge running sum and output with correction factors
        L_i_new = (tl.exp(M_i_acc - M_i_new) * L_i_acc
                   + tl.exp(M_ij - M_i_new) * tl.sum(P_ij, axis=1, keep_dims=True))

        # Cast P to match V's dtype for the matmul
        P_ij_cast = P_ij.to(V_block_ptr.type.element_ty)
        O_i_new = (tl.exp(M_i_acc - M_i_new) * O_i_acc
                   + tl.exp(M_ij - M_i_new) * tl.dot(P_ij_cast, V_j))

        # Update running accumulators for the next iteration
        M_i_acc = M_i_new
        O_i_acc = O_i_new
        L_i_acc = L_i_new

        # Advance K and V block pointers to the next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Final normalization and log-sum-exp
    O_i = O_i_acc / L_i_acc
    L_i = M_i_acc + tl.log(L_i_acc)
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # q, k, v shape: (batch_size, seq_len, d_model)
        batch_size = q.shape[0]
        Nq = q.shape[1]  # query sequence length
        Nk = k.shape[1]  # key sequence length
        d = q.shape[2]   # head dimension

        Bq = 64  # query tile size
        Bk = 64  # key tile size
        Tq = Nq // Bq

        scale = 1 / d**0.5

        O = torch.zeros_like(q)
        L = torch.zeros(batch_size, Nq, device=q.device)

        # Grid: (num_Q_tiles, batch_size)
        grid = (Tq, batch_size)

        flash_fwd_kernel[grid](q, k, v, O, L,
                               q.stride(0), q.stride(1), q.stride(2),
                               k.stride(0), k.stride(1), k.stride(2),
                               v.stride(0), v.stride(1), v.stride(2),
                               O.stride(0), O.stride(1), O.stride(2),
                               L.stride(0), L.stride(1),
                               Nq, Nk,
                               scale, d, Bq, Bk,
                               is_causal)

        ctx.save_for_backward(q, k, v, L)
        ctx.is_causal = is_causal
        return O
