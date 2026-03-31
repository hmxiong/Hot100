import math

import torch
import triton
import triton.language as tl


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@triton.jit
def elementwise_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, (a.to(tl.float32) + b.to(tl.float32)).to(a.dtype), mask=mask)


def elementwise_add(a, b, *, block: int = 1024):
    c = a.new_empty(a.shape)
    n = a.numel()
    grid = (ceil_div(n, block),)
    elementwise_add_kernel[grid](a, b, c, n_elements=n, BLOCK=block)
    return c


@triton.jit
def reduce_sum_atomic_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, s)


def reduce(x, out, *, block: int = 1024):
    n = x.numel()
    grid = (ceil_div(n, block),)
    reduce_sum_atomic_kernel[grid](x, out, n_elements=n, BLOCK=block)


@triton.jit
def reduce_v3_sum_atomic_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, s)


def recude_v3(x, out, *, block: int = 1024):
    n = x.numel()
    grid = (ceil_div(n, block),)
    reduce_v3_sum_atomic_kernel[grid](x, out, n_elements=n, BLOCK=block)


@triton.jit
def softmax_1d_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    z = tl.exp(x - m)
    denom = tl.sum(z, axis=0)
    y = z / denom
    tl.store(y_ptr + offsets, y, mask=mask)


def softmax(x, *, block: int = 1024):
    y = x.new_empty(x.shape)
    n = x.numel()
    if n > block:
        raise ValueError("softmax(x) expects x.numel() <= block")
    softmax_1d_kernel[(1,)](x, y, n_elements=n, BLOCK=block)
    return y


@triton.jit
def max_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    tl.atomic_max(out_ptr, m)


@triton.jit
def sum_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr, s)


@triton.jit
def soft_max_kernel(x_ptr, y_ptr, sum_ptr, max_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.load(max_ptr).to(tl.float32)
    s = tl.load(sum_ptr).to(tl.float32)
    y = tl.exp(x - m) / s
    tl.store(y_ptr + offsets, y, mask=mask)


def soft_max(x, max_val, sum_val, *, block: int = 1024):
    y = x.new_empty(x.shape)
    n = x.numel()
    grid = (ceil_div(n, block),)
    soft_max_kernel[grid](x, y, sum_val, max_val, n_elements=n, BLOCK=block)
    return y


@triton.jit
def transpose_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    y_offsets = offs_n[:, None] * M + offs_m[None, :]
    tl.store(y_ptr + y_offsets, tl.trans(x), mask=mask.T)


def transpose(x, M: int, N: int, *, block_m: int = 32, block_n: int = 32):
    y = x.new_empty((N, M))
    grid = (ceil_div(M, block_m), ceil_div(N, block_n))
    transpose_kernel[grid](x, y, M=M, N=N, BLOCK_M=block_m, BLOCK_N=block_n)
    return y


@triton.jit
def transpose_v2_kernel(x_ptr, y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0, cache_modifier=".ca")
    y_offsets = offs_n[:, None] * M + offs_m[None, :]
    tl.store(y_ptr + y_offsets, tl.trans(x), mask=mask.T)


def transpose_v2(x, M: int, N: int, *, block_m: int = 32, block_n: int = 32):
    y = x.new_empty((N, M))
    grid = (ceil_div(M, block_m), ceil_div(N, block_n))
    transpose_v2_kernel[grid](x, y, M=M, N=N, BLOCK_M=block_m, BLOCK_N=block_n)
    return y


@triton.jit
def sgemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        a_offsets = rm[:, None] * K + (k0 + rk)[None, :]
        b_offsets = (k0 + rk)[:, None] * N + rn[None, :]
        a = tl.load(
            a_ptr + a_offsets,
            mask=(rm[:, None] < M) & (((k0 + rk)[None, :]) < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    c_offsets = rm[:, None] * N + rn[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def sgemm(A, B, *, block_m: int = 16, block_n: int = 16, block_k: int = 32):
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError("A.shape[1] must equal B.shape[0]")
    C = A.new_empty((M, N))
    grid = (ceil_div(M, block_m), ceil_div(N, block_n))
    sgemm_kernel[grid](A, B, C, M=M, N=N, K=K, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return C


@triton.jit
def sgemm_v2_kernel(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        a_offsets = rm[:, None] * K + (k0 + rk)[None, :]
        b_offsets = (k0 + rk)[:, None] * N + rn[None, :]
        a = tl.load(a_ptr + a_offsets, mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    c_offsets = rm[:, None] * N + rn[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def sgemm_v2(A, B):
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError("A.shape[1] must equal B.shape[0]")
    C = A.new_empty((M, N))
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sgemm_v2_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    return C


@triton.jit
def sgemm_thread_tile_kernel(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        a_offsets = rm[:, None] * K + (k0 + rk)[None, :]
        b_offsets = (k0 + rk)[:, None] * N + rn[None, :]
        a = tl.load(a_ptr + a_offsets, mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    c_offsets = rm[:, None] * N + rn[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def sgemm_thread_tile(A, B):
    M, K = A.shape
    K2, N = B.shape
    if K2 != K:
        raise ValueError("A.shape[1] must equal B.shape[0]")
    C = A.new_empty((M, N))
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sgemm_thread_tile_kernel[grid](A, B, C, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=8)
    return C


@triton.jit
def softmax_matrix_kernel(x_ptr, y_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    cols = offs
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    z = tl.exp(x - m)
    denom = tl.sum(z, axis=0)
    y = z / denom
    tl.store(y_ptr + row * N + cols, y, mask=mask)


def softmax_matrix(x, *, block_n: int = 1024):
    M, N = x.shape
    y = x.new_empty((M, N))
    grid = (M,)
    softmax_matrix_kernel[grid](x, y, M=M, N=N, BLOCK_N=block_n)
    return y


@triton.jit
def flash_attn_fwd_f16_kernel(q_ptr, k_ptr, v_ptr, o_ptr, stride_qh, stride_qm, stride_qd, stride_kh, stride_kn, stride_kd, stride_vh, stride_vn, stride_vd, stride_oh, stride_om, stride_od, seqlen_q: tl.constexpr, seqlen_k: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for n0 in tl.static_range(0, seqlen_k, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + pid_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)
        v = tl.load(v_ptr + pid_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < HEAD_DIM), other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * scale
        if CAUSAL:
            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            qk = tl.where(k_pos > q_pos, -float("inf"), qk)

        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_i_new

    out = acc / l_i[:, None]
    tl.store(o_ptr + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, out.to(tl.float16), mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < HEAD_DIM))


def flash_attn_fwd_f16(q, k, v, *, causal: bool = False, block_m: int = 64, block_n: int = 64):
    B, H, SQ, D = q.shape
    _, _, SK, D2 = k.shape
    if D2 != D:
        raise ValueError("q.shape[-1] must equal k.shape[-1]")
    if v.shape != (B, H, SK, D):
        raise ValueError("v must have shape [B, H, seqlen_k, head_dim]")

    o = q.new_empty((B, H, SQ, D))
    grid = (triton.cdiv(SQ, block_m), B * H)
    q2 = q.reshape(B * H, SQ, D)
    k2 = k.reshape(B * H, SK, D)
    v2 = v.reshape(B * H, SK, D)
    o2 = o.reshape(B * H, SQ, D)
    flash_attn_fwd_f16_kernel[grid](
        q2,
        k2,
        v2,
        o2,
        q2.stride(0),
        q2.stride(1),
        q2.stride(2),
        k2.stride(0),
        k2.stride(1),
        k2.stride(2),
        v2.stride(0),
        v2.stride(1),
        v2.stride(2),
        o2.stride(0),
        o2.stride(1),
        o2.stride(2),
        seqlen_q=SQ,
        seqlen_k=SK,
        HEAD_DIM=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
        num_warps=4,
    )
    return o


@triton.jit
def topk_bitonic_rowwise_kernel(x_ptr, out_vals_ptr, out_idx_ptr, rows, cols, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_mask = row < rows
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * cols + offs, mask=row_mask & (offs < cols), other=-float("inf")).to(tl.float32)
    order = tl.argsort(x, descending=True)
    vals_sorted = tl.load(x_ptr + row * cols + order, mask=row_mask & (order < cols), other=-float("inf")).to(tl.float32)
    ranks = tl.arange(0, BLOCK)
    maskk = ranks < K
    store_ranks = tl.where(maskk, ranks, 0)
    tl.store(out_vals_ptr + row * K + store_ranks, vals_sorted, mask=row_mask & maskk)
    tl.store(out_idx_ptr + row * K + store_ranks, order.to(tl.int32), mask=row_mask & maskk)


def topk_bitonic_rowwise(x, k: int, *, block: int = 1024):
    rows, cols = x.shape
    out_vals = x.new_empty((rows, k), dtype=x.dtype)
    out_idx = x.new_empty((rows, k), dtype=torch.int32)
    grid = (rows,)
    topk_bitonic_rowwise_kernel[grid](x, out_vals, out_idx, rows, cols, K=k, BLOCK=block)
    return out_vals, out_idx


@triton.jit
def scan_inclusive_block_i32_kernel(in_ptr, out_ptr, block_sums_ptr, n_elements, WRITE_SUMS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0).to(tl.int32)
    x = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, x, mask=mask)
    if WRITE_SUMS:
        tl.store(block_sums_ptr + pid, x[BLOCK - 1])


@triton.jit
def add_scan_block_offsets_i32_kernel(out_ptr, scanned_block_sums_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    base = tl.load(scanned_block_sums_ptr + (pid - 1), mask=pid > 0, other=0).to(tl.int32)
    x = tl.load(out_ptr + offs, mask=mask, other=0).to(tl.int32)
    tl.store(out_ptr + offs, x + base, mask=mask)


def scan_inclusive_block_i32(x, *, block: int = 1024):
    n = x.numel()
    out = x.new_empty((n,), dtype=x.dtype)
    num_blocks = ceil_div(n, block)
    block_sums = x.new_empty((num_blocks,), dtype=x.dtype)
    grid = (num_blocks,)
    scan_inclusive_block_i32_kernel[grid](x, out, block_sums, n, WRITE_SUMS=True, BLOCK=block)
    if num_blocks > 1:
        if num_blocks > block:
            raise ValueError("scan_inclusive_block_i32 expects num_blocks <= block")
        scanned_block_sums = x.new_empty((num_blocks,), dtype=x.dtype)
        scan_inclusive_block_i32_kernel[(1,)](block_sums, scanned_block_sums, scanned_block_sums, num_blocks, WRITE_SUMS=False, BLOCK=block)
        add_scan_block_offsets_i32_kernel[grid](out, scanned_block_sums, n_elements=n, BLOCK=block)
    return out
