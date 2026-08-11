# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fused Triton sparse-MLA prefill for DeepSeek Sparse Attention (DSA).

Relationship to ``dsa/triton_sparse_mla.py``
--------------------------------------------
That module implements the same core form — one Triton program per query token,
online softmax held in registers, no split/merge — and reaches it for the same
reason: after TP the attention tile is tiny, so a small per-token program beats
a wide cooperative block. It is reachable only on gfx950, only for an FP8 KV
cache, and only at one pinned shape (16 heads, ``d_v`` 512, tail 64, topk 2048),
so on NVIDIA the DSA prefill never takes it.

This module is the CUDA-side sibling: bf16 (the FP8 path quantises ``P`` before
the ``PV`` product; this one does not), no shape pin, a concatenated ``q``
matching the FlashMLA entry signature, and two exact fast paths the gfx950 path
does not have. On the base path the two are within ~10% of each other on real
captured indices; the separation comes from ``union`` (see below).

Against the kernels NVIDIA DSA prefill actually dispatches to, it removes two
structural costs:

* ``flash_mla_sparse_fwd`` requires ``num_heads % 64`` (Hopper) or ``% 128``
  (Blackwell). After TP the model has far fewer heads (8 at TP8), so the head
  dim is zero-padded and 7/8 (resp. 15/16) of the tensor-core work is wasted.
* The split-decode form writes ``O(T * splits)`` partials to HBM and merges
  them in a second kernel.

This kernel runs **one program per query token** with a ``BLOCK_H=16`` head tile
(pad-into-tile, not pad-into-grid), keeps the online softmax entirely in
registers, and reuses ``V`` as ``K[:, :d_v]`` (MLA latent) so the value rows are
never gathered twice. No global partials, no merge pass.

The ``BLOCK_H`` floor was 16 on the grounds that dropping it to the exact head
count at H=8 is bitwise-identical but ~7% slower. That holds only while the tile
is held fixed: on SM120 the pair (``BLOCK_H`` 8, a narrower and deeper tile) is
1.27-1.47x faster than (16, the wide tile), because halving the ``[BLOCK_H,
d_v]`` fp32 accumulator is what buys the register budget for the deeper
pipeline. See ``_PINNED_NARROW_H``. Architectures that have not been re-measured
keep the 16-row floor.

Interface matches ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``::

    q       [T, H, 576] bf16    (absorbed MLA: 512 nope + 64 rope)
    kv      [S, 576]    bf16    (V is kv[:, :512])
    indices [T, topk]   int32   (-1 or >= S marks an invalid slot)
    out     [T, H, 512] bf16

A row of ``indices`` must not name the same KV position twice. Top-k selection
cannot, so this holds for every DSA caller; it is stated because ``union``
gathers the distinct union of G rows and weights each position once, whereas
the base path would weight a repeat twice.

Optional exact fast paths (opt-in, default off; both are algebraically
equivalent to the base path, not approximations):

``dense_prefix``
    Tokens whose top-k selects the whole causal prefix (``t + 1 <= topk``) are
    dense causal attention by definition; run FA-style tiles with zero gather
    behind an exact count/min/max guard that re-verifies the selection set.
``union``
    ``G`` adjacent query tokens share one gathered union index set; a per-row
    ownership bitmask restores the exact per-token softmax. This is where the
    win lives: on real GLM-5.1 captures the union of 4 neighbouring tokens'
    selections is only ~1.03x the size of one token's, because the indexer's
    scores move slowly in ``t``, so one gather serves four tokens. Uniformly
    random indices do not have that structure and understate the path badly —
    benchmark it on captured indices, not synthetic ones.

How the two carry over to DeepSeek-V4 (SM120, RTX 5080, measured 2026-08-10 at
T=4096, h=8, d_qk=d_v=512, combined topk=640):

* ``union`` transfers, but on a tighter margin than GLM, because DSv4 gathers
  640 rows where GLM gathers 2048 — less work to amortise the mark/compact
  passes against. Sweeping the neighbour-retention of the top-k half:

      retention   union size (G=4)   speedup
        0.50           2.02x          0.86x   <- loses
        0.75           1.54x          1.35x
        0.90           1.26x          1.62x
        0.97           1.12x          1.76x

  128 of the 640 rows are the sliding window, which shifts by exactly one
  position per query token, so G neighbours share ``128 - (G-1)`` of them
  whatever the indexer does. The other 512 are the indexer's, and its real
  retention is a property of the trained model — it is not measured here, so
  ``union`` stays opt-in for DSv4 and wants a captured-index check before it is
  turned on. (For calibration the same sweep on the GLM shape wins 1.17x even at
  a 1.88x union size, so the break-even is shape-dependent, not a fixed ratio.)

* ``dense_prefix`` does **not** transfer. DSv4's prefix region is only 640 tokens
  deep against GLM's 2048, and at that depth the dense tile is no faster than the
  per-token path: 0.98x at T=640 (100% covered), 0.89x at 1024, 0.95x at 2048,
  0.98x at 4096, 0.99x at 8192. Leave it off for DSv4; the exact-set guard makes
  enabling it harmless but pointless.

What this replaces on DeepSeek-V4 / SM120
-----------------------------------------
SM120 does not run the sparse-prefill path at all: `_forward_prefill_sparse` is
gated off (`deepseek_v4_backend.py`, "sparse_prefill_fwd does not support
SM120"), so prefill goes through `flash_mla_with_kvcache_sm120`, whose first act
is `_split_kv_pages_to_64` — a pure layout transcode from SGLang's page_size 256
pool into the page_block_size 64 that FlashInfer's DSv4 entry requires. Zero
math, paid per layer per forward.

A gather addresses tokens directly (`page = tok // page_size`), so this kernel
does not need it. Measured on RTX 5080 at DSv4's shape (topk 640, h=8, 8192-row
pool), per layer:

    T      page split alone   dequant + this kernel   ratio
    512        0.222 ms             0.177 ms          1.25x
    2048       0.856 ms             0.556 ms          1.54x
    8192       3.398 ms             2.137 ms          1.59x

The comparison is deliberately lopsided in the baseline's favour: the left
column is only the transcode, with its attention kernel still to come, while the
right column is the whole path. Even if that attention were free, routing prefill
here is 1.59x at 8192 tokens — and the transcode is per layer, so at 43 layers it
is ~146 ms of pure copying per 8192-token forward.

Where the time actually goes on DeepSeek-V4 / SM120
---------------------------------------------------
Measured on RTX 5080 (sm_120, torch 2.11+cu130 / triton 3.6), T=4096, topk=640,
S=8192 pool rows:

* **Head padding dominates, and it is not an attention-math problem.** DSv4 pads
  ``q`` from the heads TP actually leaves to 64 before the FlashMLA / FlashInfer
  entry and slices the output back (``models/deepseek_v4.py``), because that
  entry needs the head count aligned. Running those 64 padded rows through this
  kernel costs **7.35x** running the 8 real ones (10.96 ms vs 1.49 ms; 1.02x at
  h=16, 1.56x at h=32). A per-token kernel pads into a 16-row mma tile instead,
  so the padding simply does not happen — this is the single largest recoverable
  item, and it comes from *which kernel is dispatched*, not from tuning one.
* The fp8 -> bf16 dequant that feeds a gather is not worth removing: 0.0145 ms
  for the whole 8192-row pool, ~1% of the attention. See the paged-fp8 section
  below for the attempt and why it lost.
* **The gather is not the wall.** Stripping the attention math and timing the
  same gather alone gives 0.622 ms against the kernel's 1.488 ms (RTX 5080), so
  the gather is 42% of the runtime at 3.92 TiB/s out of L2 (the pool is 8 MiB
  and fits). That is why reading fewer bytes cannot pay: even a free halving of
  the gather bytes would cap out around 1.26x.
* **Nsight Compute on the retuned kernel says the tensor pipe is.** Compute (SM)
  throughput 70.38%, with Tensor the top pipeline at 70.4% ("well-utilized");
  memory 52.30%, DRAM only 6.02%, L2 hit 96.89%, no register or shared-memory
  spilling, and 31.6 of every 32 bytes per sector used. An earlier reading of
  these timings as issue-bound was wrong — it compared against the wrong tensor
  peak. Being tensor-bound is why removing padded mma rows (``_PINNED_NARROW_H``)
  paid and why adding instructions to save bytes (the paged-fp8 gather) did not.
* The remaining headroom NCU names is occupancy, not the pipes: theoretical
  16.67% / achieved 16.48%, i.e. 2 blocks per SM, limited *simultaneously* by
  registers and shared memory (Block Limit Registers = 2, Block Limit Shared Mem
  = 2), with schedulers reporting "No Eligible" 68.96%. Its estimated speedup for
  fixing occupancy is 29.6%. Both limits have to move together; the
  ``[BLOCK_H, d_v]`` fp32 accumulator is the obvious target and is already half
  what it was.
* Building the kernel up in layers over the same gather splits the rest:
  gather 56%, QK dot + online softmax 6.7%, **PV accumulate 37%** (RTX 5090,
  T=4096, h=8, topk=640). The two dots are the same FLOP count, so the PV side
  is not paying for arithmetic — it is paying for the wide ``[BLOCK_H, d_v]``
  fp32 accumulator it rescales and writes every iteration. Halving it is what
  ``_PINNED_NARROW_H`` buys, and it is worth 1.27-1.47x.

What is *not* left on the table (measured, so it need not be re-derived)
-----------------------------------------------------------------------
Nsight Compute puts the remaining headroom at occupancy — 2 blocks per SM,
limited simultaneously by registers and shared memory, estimated speedup 29.6%.
Four attempts to collect it all failed, each for a reason that is a property of
the shape rather than of the tuning. RTX 5080, DSv4 shape (d_qk=d_v=512,
topk 640, h=8), T=4096 unless stated:

* **The base-path tile is at its global optimum.** All 96 combinations of
  ``BLOCK_H`` {8,16} x ``BLOCK_N`` {16,32,64,128} x warps {2,4,8} x stages
  {1,2,3,4} were timed: the shipping ``BLOCK_H=8 (32,4,3)`` wins at 1.186 ms,
  ahead of ``(32,4,4)`` 1.198 and ``(16,2,2)`` 1.354. Nothing in the tile space
  buys the 29.6%.
* **The mma padding at h=8 cannot be removed without sharing the gather.**
  Eight real head rows sit in a 16-row mma tile, so half the tensor rows are
  padding, and transposing the dots does not help: it moves the pad from M to N,
  which Triton widens to 16 as well. Packing two tokens without sharing their
  index sets is also neutral — the combined dot is twice the size with the same
  useful fraction. Sharing the gather is the only lever, i.e. ``union``.
* **The gather does not become the wall at production pool sizes.** Every other
  measurement here used an 8 MiB pool that fits in this device's 64 MiB L2 (NCU
  confirmed, L2 hit 96.89%), so the obvious worry is that the ranking inverts
  once the pool leaves cache. It does not: sweeping the pool from 8 MiB to
  2 GiB at T=2048 costs only 1.31x (0.570 -> 0.747 ms, effective 2193 -> 1674
  GiB/s). DSA's own locality is why — the sliding-window half shifts one
  position per token, so consecutive tokens re-read almost all of it.
* **``union``'s break-even is not overhead, and not occupancy either.** The
  mark and compact launches together are 4-7% of union's runtime at retention
  >= 0.5 (0.008-0.020 ms and 0.038-0.067 ms against ~1.1-1.3 ms), so fusing
  them cannot move break-even. Occupancy looked more promising — the union
  kernel takes 84.7/67.8 KB of shared memory against the base path's 40.8 KB,
  i.e. 1 block per SM against 2 — but forcing it under half the budget makes it
  *slower*: at retention 0.50 the configs that reach 3 blocks/SM (33.4 KB) run
  0.86-0.88x base, behind the 1-block/SM ``(64,4,2)`` at 0.896x. Across 162
  (G, BLOCK_N, warps, stages) combinations the shipping ``G=4 (32,4,2)`` is
  optimal at every retention where union pays at all: 1.070x at 0.75 and 1.247x
  at 0.90. The wider mma tile is worth more than the extra block.

So the base path is at its ceiling on this shape, and ``union`` remains the only
lever — gated not on tuning but on the trained indexer's neighbour retention,
which needs a real capture to measure. ``G=2`` is not a shortcut to it: despite
``G*H = 16`` filling the mma tile exactly at h=8, it loses at every retention
below 0.90 (0.68x at 0.00, 0.86x at 0.50, 1.045x at 0.97) because two tokens
amortise the union's fixed work half as well as four.

One anomaly is recorded rather than explained: at retention 0.25 the union
*launcher* (not its kernels) cost 6.2 ms, 83% of the call, for both G=2 and
G=4, against 0.06-0.08 ms at every higher retention. It sits in the host-synced
``amin``/``amax`` plus workspace management, not in the three Triton launches,
which were timed separately and were normal. ``union`` is opt-in and off by
default, so this is not on any shipped path, but it wants a look before it is
turned on.

Decode
------
Nothing here is prefill-only: there is no causal structure and, on the base
path, no cross-token sharing, so a decode step is simply ``T = batch``. It works
and it is the faster of the two available kernels — 0.0324 ms at batch 1 against
SGLang's SM120 ``flash_mla_sparse_decode_triton`` at 0.211 ms — but the design
is the wrong shape for small-batch decode and the numbers say so: one program
per query token means batch 1 lights up 1 of 84 SMs, and the wall time is flat
at ~0.033 ms from batch 1 to 64 (32.4 us/token down to 0.53), only reaching its
0.267 us/token floor at batch >= 2048. Filling the device at small batch needs
splitting one token's top-k across CTAs and merging — exactly the partials-and-
merge structure this kernel exists to avoid. Use it for decode because it wins
today, not because the shape suits it.

All tunables are explicit arguments — the module reads no environment
variables. If a tuned tile exceeds the device shared-memory budget (e.g. a
large head count on SM120's 100 KB), the launcher steps down through smaller
tiles instead of raising ``OutOfResources``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Swept on the hardware named; anything else gets _UNTUNED_DEFAULT and a warning.
_PINNED = {
    (9, 0): (64, 8, 2),  # SM90, swept at T=8192
    (12, 0): (64, 4, 2),  # SM120, swept at T=8192
    (12, 1): (64, 4, 2),
}
# The SM120 entry was swept on the GLM shape (d_qk=576, topk=2048). Re-swept on
# DeepSeek-V4's (d_qk=d_v=512, topk=640, T=8192, h=8): (64, 4, 2) is still the
# optimum -- 27 configs, the runner-up (128, 4, 2) is 0.995x and the best is
# 1.004x, both inside the 17 us run-to-run sigma. No DSv4-specific entry needed.
_UNTUNED_DEFAULT = (64, 8, 3)


@triton.jit
def _nsa_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    o_ptr,
    scale_ptr,  # FP8 only: [s_qk = qs*ks, s_k = ks] fp32; unused when not FP8
    sink_ptr,  # [H] fp32 learned per-head sink logit; unused when not HAS_SINK
    sm_scale,
    topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    D_V_PAD: tl.constexpr,  # next power of two >= D_V; tl.arange needs one
    BLOCK_N: tl.constexpr,
    FP8: tl.constexpr,
    MATH_BF16: tl.constexpr,  # with FP8: cast tiles to bf16 after load (halve L1/L2
    # gather bytes, keep the SM90-fast bf16 mma path)
    IDX64: tl.constexpr,  # int64 row addressing only when the KV pool can overflow
    # int32*D_QK (rows > ~3.7M). int32 fast path: SASS drops
    # 36x IMAD.WIDE -> IMAD in the hot gather loop
    # (cuda-agent step-001, -97us z=8.11 @ T=8192).
    HAS_SINK: tl.constexpr,  # DeepSeek-V4: a learned per-head logit joins the
    # softmax denominator but contributes no value row.
):
    t = tl.program_id(0)
    D_TAIL: tl.constexpr = D_QK - D_V

    if FP8:
        # inputs pre-scaled by 448/amax in the wrapper; undo inside the math:
        # qk_real = qk_fp8 * qs*ks/448^2 ; pv_real = pv_fp8 * ks/448^2 (P carries x448)
        qk_scale = sm_scale * tl.load(scale_ptr) / (448.0 * 448.0)
        out_scale = tl.load(scale_ptr + 1) / (448.0 * 448.0)
    else:
        qk_scale = sm_scale
        out_scale = 1.0

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    dv = tl.arange(0, D_V_PAD)
    # tl.arange cannot express a non-power-of-two value dim, so carry the tile at
    # the next power of two and mask the surplus columns to zero: they then
    # contribute nothing to either dot. When D_V is already a power of two the
    # mask is all-true and the generated code is unchanged.
    vmask = dv < D_V

    qb = q_ptr + t * H * D_QK
    q_main = tl.load(
        qb + h[:, None] * D_QK + dv[None, :],
        mask=hmask[:, None] & vmask[None, :],
        other=0.0,
    )
    if FP8 and MATH_BF16:
        q_main = q_main.to(tl.bfloat16)
    # DeepSeek-V4 has no rope tail to carry separately: its 512-wide head is both
    # the key and the whole value (the rope columns are un-rotated downstream, on
    # the attention output), so D_QK == D_V and the second dot disappears.
    # tl.arange(0, 0) is a compile error, so the tail must be elided, not masked.
    if D_TAIL > 0:
        dt = tl.arange(0, D_TAIL)
        q_tail = tl.load(
            qb + h[:, None] * D_QK + (D_V + dt)[None, :], mask=hmask[:, None], other=0.0
        )
        if FP8 and MATH_BF16:
            q_tail = q_tail.to(tl.bfloat16)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V_PAD], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = tl.load(len_ptr + t)
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        valid = idx >= 0
        if IDX64:
            row = tl.where(valid, idx, 0).to(tl.int64)
        else:
            row = tl.where(valid, idx, 0)
        kb = kv_ptr + row[:, None] * D_QK
        kv_main = tl.load(
            kb + dv[None, :], mask=valid[:, None] & vmask[None, :], other=0.0
        )
        if FP8 and MATH_BF16:
            kv_main = kv_main.to(tl.bfloat16)

        qk = tl.dot(q_main, tl.trans(kv_main))
        if D_TAIL > 0:
            kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)
            if FP8 and MATH_BF16:
                kv_tail = kv_tail.to(tl.bfloat16)
            qk = tl.dot(q_tail, tl.trans(kv_tail), qk)
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if FP8:
            p_q = (p * 448.0).to(kv_main.dtype)
        else:
            p_q = p.to(kv_main.dtype)
        acc = acc * alpha[:, None] + tl.dot(p_q, kv_main)
        m_i = m_new

    if HAS_SINK:
        # The sink is a raw logit (already in natural-log space, not scaled by
        # sm_scale) that joins the denominator without contributing a value row,
        # matching `_apply_attn_sink`'s logaddexp(lse, sink) in the SM120 decode
        # path. `acc` and `l_i` are both held at base `m_base`, so rescaling the
        # pair to a combined max leaves acc/l_i unchanged while keeping
        # exp(sink - base) from overflowing when every logit is far below sink.
        sink = tl.load(sink_ptr + h, mask=hmask, other=-float("inf")).to(tl.float32)
        m_base = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_comb = tl.maximum(m_base, sink)
        rescale = tl.exp(m_base - m_comb)
        l_i = l_i * rescale + tl.exp(sink - m_comb)
        acc = acc * rescale[:, None]

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (out_scale / l_safe[:, None])
    tl.store(
        o_ptr + t * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hmask[:, None] & vmask[None, :],
    )


# ---------------------------------------------------------------------------
# Dense-prefix fast path (opt-in: GLM_NSA_DENSE_PREFIX=1). DSA semantics: token
# t with t+1 <= topk selects its ENTIRE prefix -> exact dense causal attention,
# FA2-tiled (M rows 100% real, zero gather). Guarded by an exact set check
# (count+min+max pin the selected set to base+{0..t} under unique indices) and
# rebased for pool-row offsets. Evidence: docs/dense-prefix-report.md (1.84x on
# the prefix region; +12%/+5% total at 4k/8k; whole-request 1.84x for T<=2048).
# ---------------------------------------------------------------------------


@triton.jit
def _dense_prefix_kernel(
    q_ptr,
    kv_ptr,
    o_ptr,
    sink_ptr,  # [H] fp32; unused when not HAS_SINK
    sm_scale,
    P,
    H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    D_TAIL: tl.constexpr = D_QK - D_V

    m0 = pid_m * BLOCK_M
    m = tl.arange(0, BLOCK_M)
    rows = m0 + m
    rmask = rows < P
    dv = tl.arange(0, D_V)

    qb = q_ptr + (rows * H + h).to(tl.int64)[:, None] * D_QK
    q_main = tl.load(qb + dv[None, :], mask=rmask[:, None], other=0.0)
    # DeepSeek-V4 has no rope tail (D_QK == D_V); tl.arange(0, 0) does not compile,
    # so the tail is elided at trace time rather than masked.
    if D_TAIL > 0:
        dt = tl.arange(0, D_TAIL)
        q_tail = tl.load(qb + (D_V + dt)[None, :], mask=rmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D_V], tl.float32)
    n = tl.arange(0, BLOCK_N)

    # phase-1 may only cover WHOLE blocks strictly below m0; the remainder
    # [safe_lo, hi) goes through the causal-masked phase-2 (any BLOCK_M/BLOCK_N).
    safe_lo = (m0 // BLOCK_N) * BLOCK_N
    for k0 in tl.range(0, safe_lo, BLOCK_N):  # fully-unmasked blocks
        kb = kv_ptr + (k0 + n).to(tl.int64)[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :])
        qk = tl.dot(q_main, tl.trans(kv_main))
        if D_TAIL > 0:
            kv_tail = tl.load(kb + (D_V + dt)[None, :])
            qk = tl.dot(q_tail, tl.trans(kv_tail), qk)
        qk = qk * sm_scale
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    hi = tl.minimum(m0 + BLOCK_M, P)
    for k0 in tl.range(safe_lo, hi, BLOCK_N):  # causal boundary blocks
        cmask = (k0 + n) < P
        kb = kv_ptr + tl.where(cmask, k0 + n, 0).to(tl.int64)[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :], mask=cmask[:, None], other=0.0)
        qk = tl.dot(q_main, tl.trans(kv_main))
        if D_TAIL > 0:
            kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=cmask[:, None], other=0.0)
            qk = tl.dot(q_tail, tl.trans(kv_tail), qk)
        qk = qk * sm_scale
        causal = (k0 + n)[None, :] <= rows[:, None]
        qk = tl.where(causal & cmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    if HAS_SINK:
        # One head per program, so the sink is a scalar here. Same combined-max
        # rescale as the base path: acc and l_i are both held at m_base.
        sink = tl.load(sink_ptr + h).to(tl.float32)
        m_base = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_comb = tl.maximum(m_base, sink)
        rescale = tl.exp(m_base - m_comb)
        l_i = l_i * rescale + tl.exp(sink - m_comb)
        acc = acc * rescale[:, None]

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    ob = o_ptr + (rows * H + h).to(tl.int64)[:, None] * D_V
    tl.store(ob + dv[None, :], acc.to(o_ptr.dtype.element_ty), mask=rmask[:, None])


def _dense_prefix_path(
    q, kv, indices, sm_scale, d_v, out, dense_config=None, attn_sink=None
):
    """Handle rows [0, P) densely when they provably selected their full prefix.
    Returns P (rows handled; 0 = not applicable). Exact-set gate: with unique
    indices, count==t+1 & min==base & max==base+t pin the set to base+{0..t}."""
    T, h, d_qk = q.shape
    topk = indices.shape[-1]
    P = min(topk, T)
    if P < 128:
        return 0
    pre = indices[:P]
    valid = pre >= 0
    counts = valid.sum(dim=-1, dtype=torch.int32)
    want = torch.arange(P, dtype=torch.int32, device=indices.device)
    vmax = pre.amax(dim=-1)
    vmin = torch.where(valid, pre, pre.new_full((), 2**31 - 1)).amin(dim=-1)
    base = vmin[0]
    ok = (counts == want + 1).all() & (vmin == base).all() & (vmax == base + want).all()
    if not bool(ok):
        return 0
    b = int(base)
    if b + P > kv.shape[0]:
        return 0
    if dense_config is not None:
        bm, bn, warps, stages = dense_config
    else:
        _cap = torch.cuda.get_device_capability()
        # SM90: 228KB smem; SM120: 99KB -> smaller tiles (validated on-box 2026-07-22)
        bm, bn, warps, stages = (32, 64, 4, 2) if _cap[0] == 9 else (16, 32, 8, 2)
    _dense_prefix_kernel[(triton.cdiv(P, bm), h)](
        q[:P],
        kv[b:],
        out[:P],
        attn_sink if attn_sink is not None else out,  # unread placeholder
        sm_scale,
        P,
        H=h,
        D_QK=d_qk,
        D_V=d_v,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=warps,
        num_stages=stages,
        HAS_SINK=attn_sink is not None,
    )
    return P


# ---------------------------------------------------------------------------
# v3 union tiling (opt-in: GLM_NSA_UNION=2|4). G adjacent tokens share one
# deduplicated KV gather; M = G*H rows all-real. Exact math (per-token -inf
# masking); evidence + gates in docs/decision-ledger.md v3 entries.
# ---------------------------------------------------------------------------

# Scratch for the union path, keyed by (group size, KV-span bucket, device) —
# deliberately NOT by the group count, which changes with every batch shape and
# would let the cache grow without bound over a server's lifetime. Buffers grow
# in place when a larger batch arrives, so the entry count stays bounded by the
# span buckets a model can reach.
_UNION_WS = {}
_UNION_SPAN_BUDGET = 512 << 20  # bytes for the [NG, span] int32 mark map


@triton.jit
def _union_mark_kernel(
    idx_ptr, map_ptr, K, S, base, epoch, G: tl.constexpr, BLOCK: tl.constexpr
):
    # Bytemap lanes are G-wide (not fixed 4) and carry an epoch value instead of
    # a sticky 1: stale bytes from earlier layers never match the current epoch,
    # so the compact pass needs no self-zeroing store (halves its traffic).
    pid = tl.program_id(0)
    g = pid // G
    tok = pid % G
    b = idx_ptr + pid.to(tl.int64) * K
    n = tl.arange(0, BLOCK)
    ep8 = tl.full([BLOCK], 0, tl.int8) + epoch
    for k0 in tl.range(0, K, BLOCK):
        v = tl.load(b + k0 + n, mask=(k0 + n) < K, other=-1)
        v = v - base
        valid = (v >= 0) & (v < S)
        addr = (g.to(tl.int64) * S + tl.where(valid, v, 0).to(tl.int64)) * G + tok
        tl.store(map_ptr + addr, ep8, mask=valid)


@triton.jit
def _union_compact_kernel(
    map_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    S,
    U_CAP,
    epoch,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
    STAGES: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BLOCK)
    cursor = tl.zeros([], tl.int32)
    for s0 in tl.range(0, S, BLOCK, num_stages=STAGES):
        inb = (s0 + n) < S
        w = tl.load(map_ptr + g * S + s0 + n, mask=inb, other=0).to(tl.int32)
        bits = ((w & 255) == epoch).to(tl.int32)
        bits |= (((w >> 8) & 255) == epoch).to(tl.int32) * 2
        if LANES == 4:
            bits |= (((w >> 16) & 255) == epoch).to(tl.int32) * 4
            bits |= (((w >> 24) & 255) == epoch).to(tl.int32) * 8
        present = (bits != 0) & inb
        wpos = cursor + tl.cumsum(present.to(tl.int32), axis=0) - present.to(tl.int32)
        tl.store(uidx_ptr + g * U_CAP + wpos, (s0 + n).to(tl.int32), mask=present)
        tl.store(ubits_ptr + g * U_CAP + wpos, bits, mask=present)
        cursor += tl.sum(present.to(tl.int32))
    tl.store(ulen_ptr + g, cursor)


@triton.jit
def _nsa_prefill_union_kernel(
    q_ptr,
    kv_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    o_ptr,
    sink_ptr,  # [H] fp32; unused when not HAS_SINK
    sm_scale,
    U_CAP,
    base,
    H: tl.constexpr,
    G: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    g = tl.program_id(0)
    D_TAIL: tl.constexpr = D_QK - D_V
    GH: tl.constexpr = G * H

    r = tl.arange(0, GH)
    tok_of_row = r // H
    dv = tl.arange(0, D_V)

    qb = q_ptr + g.to(tl.int64) * GH * D_QK
    q_main = tl.load(qb + r[:, None] * D_QK + dv[None, :])
    # DeepSeek-V4 has no rope tail (D_QK == D_V) -- elide it at trace time.
    if D_TAIL > 0:
        dt = tl.arange(0, D_TAIL)
        q_tail = tl.load(qb + r[:, None] * D_QK + (D_V + dt)[None, :])

    m_i = tl.full([GH], -float("inf"), tl.float32)
    l_i = tl.zeros([GH], tl.float32)
    acc = tl.zeros([GH, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    u_len = tl.load(ulen_ptr + g)
    ub = g.to(tl.int64) * U_CAP
    for k0 in tl.range(0, u_len, BLOCK_N):
        inb = (k0 + n) < u_len
        uidx = tl.load(uidx_ptr + ub + k0 + n, mask=inb, other=-1)
        bits = tl.load(ubits_ptr + ub + k0 + n, mask=inb, other=0)
        valid = uidx >= 0
        row = (tl.where(valid, uidx, 0) + base).to(tl.int64)
        kb = kv_ptr + row * D_QK
        kv_main = tl.load(kb[:, None] + dv[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q_main, tl.trans(kv_main))
        if D_TAIL > 0:
            kv_tail = tl.load(
                kb[:, None] + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
            )
            qk = tl.dot(q_tail, tl.trans(kv_tail), qk)
        qk = qk * sm_scale
        sel = ((bits[None, :] >> tok_of_row[:, None]) & 1) != 0
        qk = tl.where(sel & valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    if HAS_SINK:
        # Rows are (token, head) pairs laid out head-fastest, so the sink lane is
        # r % H. Same combined-max rescale as the base path.
        sink = tl.load(sink_ptr + (r % H)).to(tl.float32)
        m_base = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_comb = tl.maximum(m_base, sink)
        rescale = tl.exp(m_base - m_comb)
        l_i = l_i * rescale + tl.exp(sink - m_comb)
        acc = acc * rescale[:, None]

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + g.to(tl.int64) * GH * D_V + r[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


def _union_path(
    q, kv, indices, sm_scale, d_v, out, G, union_config=None, attn_sink=None
):
    """Returns True if handled. Budget-gated; tail rows (T % G) fall back."""
    T, h, d_qk = q.shape
    K = indices.shape[-1]
    while G > 1 and G * h > 32:  # keep the M tile within smem (configs tuned to GH<=32)
        G //= 2
    if G < 2:
        return False
    T_main = (T // G) * G
    if T_main == 0:
        return False
    # single fused reduction + ONE host sync (amax is -1-safe; amin masks -1 to INT_MAX)
    vmin_t = torch.where(
        indices >= 0,
        indices,
        torch.tensor(2**31 - 1, dtype=indices.dtype, device=indices.device),
    ).amin()
    vmax_t = indices.amax()
    vmin, vmax = torch.stack([vmin_t, vmax_t]).tolist()
    if vmax < 0:
        return False
    span = vmax - vmin + 1
    NG = T_main // G
    if NG * span * G > _UNION_SPAN_BUDGET:
        return False

    span_alloc = ((span + 4095) // 4096) * 4096
    key = (G, span_alloc, G * K, q.device)
    bufs = _UNION_WS.get(key)
    if bufs is None or bufs[0].shape[0] < NG:
        # Grow to this batch's group count; the old buffers (if any) are dropped
        # and their epoch counter restarts, which the zero-init below re-bases.
        bufs = (
            torch.zeros(NG, span_alloc * G, dtype=torch.int8, device=q.device),
            torch.empty(NG, G * K, dtype=torch.int32, device=q.device),
            torch.empty(NG, G * K, dtype=torch.int32, device=q.device),
            torch.empty(NG, dtype=torch.int32, device=q.device),
            [0],
        )
        _UNION_WS[key] = bufs
    ws_all, uidx, ubits, ulen, ep_box = bufs
    epoch = ep_box[0] + 1
    if epoch > 127:  # int8 epoch wrap: one cheap memset per 127 reuses
        # Zero every row, not just this batch's: a later, larger batch would
        # otherwise read marks left by an earlier cycle as belonging to it.
        ws_all.zero_()
        epoch = 1
    ep_box[0] = epoch
    ws, uidx, ubits, ulen = ws_all[:NG], uidx[:NG], ubits[:NG], ulen[:NG]
    U_CAP = G * K
    idx_main = indices[:T_main].contiguous()
    _union_mark_kernel[(NG * G,)](
        idx_main, ws, K, span_alloc, vmin, epoch, G=G, BLOCK=1024, num_warps=4
    )
    wsw = ws.view(torch.int16) if G == 2 else ws.view(torch.int32)
    _union_compact_kernel[(NG,)](
        wsw,
        uidx,
        ubits,
        ulen,
        span_alloc,
        U_CAP,
        epoch,
        BLOCK=1024,
        LANES=G,
        STAGES=4,
        num_warps=4,
    )
    if union_config is not None:
        bn, warps, stages = union_config
    elif torch.cuda.get_device_capability(q.device)[0] >= 12:
        # SM120 on-box sweeps: G=2 winner (64,4,3) 5.21 vs 5.50; G=4 winner (32,4,2)
        # 3.605 ms on real indices (BN=64 OORs >=115KB with the GH=32 Q tile; BN=32
        # restores the fit and the M=32 x N=32 tile beats every neighbor by >=12%).
        bn, warps, stages = (64, 4, 3) if G == 2 else (32, 4, 2)
    else:
        bn, warps, stages = (64, 4, 2) if G == 4 else (64, 8, 2)
    # The union Q tile is H*G rows, so its shared-memory footprint grows with
    # the head count: 16 heads at G=2 already exceeds SM120's 100 KB with the
    # tuned tile. Step down as the per-token launcher does rather than failing
    # the request.
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_union_kernel[(NG,)](
                q[:T_main],
                kv,
                uidx,
                ubits,
                ulen,
                out[:T_main],
                attn_sink if attn_sink is not None else ulen,  # unread placeholder
                sm_scale,
                U_CAP,
                vmin,
                H=h,
                G=G,
                D_QK=d_qk,
                D_V=d_v,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
                HAS_SINK=attn_sink is not None,
            )
            break
        except triton.runtime.errors.OutOfResources:
            continue
    else:
        return False  # no tile fits; caller falls through to the per-token path
    if T_main < T:  # tail rows through the per-token kernel
        sparse_mla_prefill(
            q[T_main:],
            kv,
            indices[T_main:],
            sm_scale,
            d_v,
            attn_sink=attn_sink,
            out=out[T_main:],
            union=False,
        )
    return True


def _topk_length(indices, topk):
    valid = indices >= 0
    any_valid = valid.any(dim=-1)
    last = topk - torch.flip(valid, [-1]).int().argmax(dim=-1)
    return torch.where(any_valid, last, torch.zeros_like(last)).to(torch.int32)


_UNTUNED_ARCH_WARNED = set()


# A large head count changes which tile wins, so it gets its own entry rather
# than riding the h<=32 one. Swept on SM120 at DSv4's shape (T=4096, topk=640):
# at h=64 the pinned (64, 4, 2) runs 10.87 ms and (16, 8, 2) runs 9.29 ms, 1.17x;
# at h<=32 the pinned tile is already the winner (h=32's best is 1.01x, inside
# noise), so those are left alone. This matters for TP1/TP2, where 64/32 heads
# are real, and for any caller that still pads its head count up.
_PINNED_WIDE_H = {
    (12, 0): (16, 8, 2),
    (12, 1): (16, 8, 2),
}
_WIDE_H = 32  # head counts above this take the wide-H entry

# A head count at or below this gets BLOCK_H = the head count instead of the
# 16-row floor, together with its own tile. The two only pay together: at the
# 16-row floor BLOCK_H changes nothing (1.00-1.02x), and at BLOCK_H 16 the wide
# tile is still the winner -- it is the pair that moves.
#
# Why it pays here: with 8 heads the 16-row tile spends half its mma rows on
# masked padding, and the fp32 accumulator it carries is [BLOCK_H, d_v] = 32 KB
# of registers. Halving both frees enough register budget for a deeper pipeline
# (num_stages 3) at a narrower BLOCK_N. That matters because the PV accumulate,
# not the gather, is where the time goes: decomposing the kernel over the same
# gather gives gather 56%, QK dot + online softmax 6.7%, PV accumulate 37%
# (RTX 5090, T=4096, h=8, topk=640).
#
# Paired same-session A/B, (BLOCK_H 16 + pinned tile) vs (BLOCK_H 8 + this one),
# RTX 5090, h=8:
#     DSv4 512/512 topk 640   T=2048/4096/8192 -> 1.27x / 1.30x / 1.36x
#     GLM  576/512 topk 2048  T=2048/4096/8192 -> 1.32x / 1.45x / 1.47x
# sigma 0.4-13.7 us against gaps of 80-1800 us.
#
# This contradicts the older note that BLOCK_H 8 is ~7% slower. That note is kept
# in spirit as the reason for the floor on architectures where it has not been
# re-measured: only entries listed here lower it, so SM90 and any untuned device
# keep exactly the previous behaviour.
_PINNED_NARROW_H = {
    (12, 0): (32, 4, 3),
    (12, 1): (32, 4, 3),
}
_NARROW_H = 8  # head counts at or below this take the narrow-H entry


def _narrow_h(device, num_heads):
    """Whether this device+head count has a measured sub-16 BLOCK_H entry."""
    return (
        num_heads is not None
        and num_heads <= _NARROW_H
        and torch.cuda.get_device_capability(device) in _PINNED_NARROW_H
    )


def _block_h(device, num_heads):
    """Rows of the head mma tile. 16 unless this device+head count was swept
    below it -- an unswept arch must not silently change tile."""
    floor = 8 if _narrow_h(device, num_heads) else 16
    return max(floor, triton.next_power_of_2(num_heads))


def _config(device, num_heads=None):
    """Per-arch tuned (BLOCK_N, num_warps, num_stages); see _PINNED."""
    cap = torch.cuda.get_device_capability(device)
    if num_heads is not None and num_heads > _WIDE_H and cap in _PINNED_WIDE_H:
        return _PINNED_WIDE_H[cap]
    if _narrow_h(device, num_heads):
        return _PINNED_NARROW_H[cap]
    if cap in _PINNED:
        return _PINNED[cap]
    if cap not in _UNTUNED_ARCH_WARNED:
        _UNTUNED_ARCH_WARNED.add(cap)
        # The kernel is correct on any SM90+ device, but the tile was only swept
        # on the architectures in _PINNED. Say so rather than quietly running a
        # config nobody has measured — add an entry there once swept.
        logger.warning(
            "triton_sparse_mla: no tuned tile for sm_%d%d; falling back to %s. "
            "Sweep and add it to _PINNED for best throughput.",
            cap[0],
            cap[1],
            _UNTUNED_DEFAULT,
        )
    return _UNTUNED_DEFAULT


def _smem_fallbacks(bn, stages):
    """Ordered (BLOCK_N, num_stages) candidates: the tuned config first, then
    progressively smaller smem footprints. Lets one pinned config serve head
    counts / devices whose smem budget the tuned tile would exceed."""
    seen, out = set(), []
    for cand in (
        (bn, stages),
        (bn, 2),
        (bn // 2, stages),
        (bn // 2, 2),
        (bn // 4, 2),
        (16, 2),
        (bn // 2, 1),
        (16, 1),
    ):
        b, ns = max(16, cand[0]), max(1, cand[1])
        if (b, ns) not in seen:
            seen.add((b, ns))
            out.append((b, ns))
    return out


def sparse_mla_prefill(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    *,
    attn_sink=None,
    topk_length=None,
    out=None,
    union=0,
    dense=False,
    config=None,
    union_config=None,
    dense_config=None,
    int64_indexing=None,
    block_h=None,
):
    """Fused sparse-MLA prefill. Returns ``out`` ``[T, H, d_v]`` bf16.

    Args:
        q: ``[T, H, d_qk]`` bf16 query (absorbed MLA; ``d_qk = d_v + rope``).
        kv: ``[S, d_qk]`` or ``[S, 1, d_qk]`` bf16 latent cache; ``V`` is
            ``kv[:, :d_v]`` (no separate value gather).
        indices: ``[T, topk]`` or ``[T, 1, topk]`` int32 selected slots; ``-1``
            or ``>= S`` marks an invalid slot and is skipped.
        sm_scale: softmax scale.
        d_v: value head dim (512 for DSA). ``d_qk == d_v`` is allowed and means
            there is no rope tail to score separately — DeepSeek-V4's 512-wide
            head is both the key and the whole value.
        attn_sink: optional ``[H]`` fp32 learned per-head sink logit (DeepSeek-V4).
            Joins the softmax denominator without contributing a value row, i.e.
            ``logaddexp(lse, sink)``. Supported on all three paths.
        topk_length: optional ``[T]`` int32 per-row valid count. Computed from
            ``indices`` when omitted; pass it to skip that reduction.
        out: optional preallocated ``[T, H, d_v]`` bf16 output.
        union: 0 (off), 2 or 4 — share one gathered union index set across ``G``
            adjacent query tokens. Exact, not an approximation: an ownership
            bitmask restores each token's own softmax support.
        dense: enable the dense-prefix identity fast path for tokens whose top-k
            covers the whole causal prefix. Guarded by an exact set check; falls
            back to the sparse path if the guard fails.
        config / union_config / dense_config: optional tile overrides
            ``(BLOCK_N, num_warps, num_stages)`` (dense takes ``(BM, BN, warps,
            stages)``). Defaults are the per-arch tuned entries in ``_PINNED``.
    """
    if kv.dim() == 3:  # [S, 1, D] -> [S, D]
        assert kv.shape[1] == 1
        kv = kv.squeeze(1)
    if indices.dim() == 3:  # [T, 1, K] -> [T, K]
        assert indices.shape[1] == 1
        indices = indices.squeeze(1)
    T, h, d_qk = q.shape
    topk = indices.shape[-1]
    if d_qk < d_v:
        raise ValueError(f"d_v ({d_v}) cannot exceed the query width ({d_qk}).")
    if attn_sink is not None:
        if attn_sink.shape != (h,):
            raise ValueError(
                f"attn_sink must be [num_heads] = [{h}]; got "
                f"{tuple(attn_sink.shape)}."
            )
        attn_sink = attn_sink.to(torch.float32).contiguous()
    if (union or dense) and (d_v & (d_v - 1)):
        # The base path carries a padded value tile, but the union and
        # dense-prefix kernels still index the value dim directly, so a
        # non-power-of-two d_v would silently mis-tile there. (DeepSeek-V4 is not
        # such a model: sglang hands it d_v = head_dim = 512, a power of two with
        # a zero-width tail, so both fast paths are open to it -- the 448 case is
        # spare capacity, not the DSv4 one.)
        raise ValueError(
            f"the union and dense-prefix fast paths need a power-of-two d_v; "
            f"got {d_v}. Run the base path for this model."
        )

    q, kv, indices = q.contiguous(), kv.contiguous(), indices.contiguous()
    if out is None:
        out = torch.empty(T, h, d_v, dtype=torch.bfloat16, device=q.device)

    if dense:
        P = _dense_prefix_path(
            q, kv, indices, sm_scale, d_v, out, dense_config, attn_sink=attn_sink
        )
        if P >= T:
            return out
        if P > 0:  # remainder continues through the normal (or union) path below
            sparse_mla_prefill(
                q[P:],
                kv,
                indices[P:],
                sm_scale,
                d_v,
                attn_sink=attn_sink,
                out=out[P:],
                union=union,
                dense=False,
            )
            return out

    if union in (2, 4) and _union_path(
        q, kv, indices, sm_scale, d_v, out, union, union_config, attn_sink=attn_sink
    ):
        return out

    if topk_length is None:
        topk_length = _topk_length(indices, topk)

    q_in, kv_in = q, kv
    scales = torch.ones(2, dtype=torch.float32, device=q.device)

    bn, warps, stages = config or _config(q.device, h)
    # int32 gather addressing unless the pool could overflow int32 element offsets
    # (row*d_qk + d_qk-1 must fit in int32) — production pools can exceed this.
    # The threshold is ~3.7M rows at d_qk=576, which no test can allocate, so the
    # mode is overridable to keep the int64 path reachable from a test.
    if int64_indexing is None:
        idx64 = kv.shape[0] > (2**31 - 1 - (d_qk - 1)) // d_qk
    else:
        idx64 = bool(int64_indexing)
    block_h = block_h or _block_h(q.device, h)
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_kernel[(T,)](
                q_in,
                kv_in,
                indices,
                topk_length,
                out,
                scales,
                # `scales` doubles as an unread placeholder so the pointer arg is
                # always a real tensor; HAS_SINK gates every load from it.
                attn_sink if attn_sink is not None else scales,
                sm_scale,
                topk,
                H=h,
                BLOCK_H=block_h,
                D_QK=d_qk,
                D_V=d_v,
                D_V_PAD=triton.next_power_of_2(d_v),
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
                FP8=False,
                MATH_BF16=False,
                IDX64=idx64,
                HAS_SINK=attn_sink is not None,
            )
            return out
        except triton.runtime.errors.OutOfResources:
            # A larger head count (BLOCK_H) or a smaller smem budget can push the
            # pinned tile over the device limit (e.g. h=32 on SM120's 100 KB).
            # Step down the K tile / pipeline depth instead of failing the request.
            continue
    raise triton.runtime.errors.OutOfResources(
        0, 0, "shared memory: no fallback config fits this device/shape"
    )


# ---------------------------------------------------------------------------
# Paged FP8 gather (DeepSeek-V4). The kernel above takes a flat bf16 KV block,
# which on DSv4 only exists because something dequantized the pool into it
# first. This one gathers straight out of the paged fp8 pools and decodes each
# row in registers, so that intermediate never has to exist.
#
# The idea was that it removes passes rather than instructions: a gathered row
# costs 584 B here against 584 (read fp8) + 1024 (write bf16) + 1024 (read bf16)
# for the materialized form. That accounting is right and the conclusion is still
# wrong -- see the measurement below.
#
# Two sources, because that is what a DSv4 layer attends over: the sliding
# window lives in the SWA pool (page_size 256) and, on compressed layers, the
# selected rows live in a separate compressed pool at a different page size
# (64 for compress_ratio 4, 2 for 128). Feeding both to one kernel keeps the
# online softmax in registers across them -- concatenating them into one buffer
# first is exactly the materialization this kernel exists to avoid.
#
# MEASURED: this is slower than gathering pre-decoded bf16, and is NOT the
# default. On SM120 (RTX 5080, T=4096, h=8, topk=640, S=8192): the bf16 gather
# runs 1.487 ms; the best of a 36-config sweep here is 3.465 ms at (BLOCK_N=32,
# warps=4, stages=1), i.e. 2.33x slower. Tuning recovered half the gap (the
# pinned stages=2/3 cost 8.2 ms -- the extra staging tiles thrash the pipeline),
# the rest is structural: it replaces one coalesced 1024-byte row read with three
# scattered reads of 448 B fp8 + 128 B rope + 8 B scale, and the extra memory
# instructions and sector waste cost more than the 440 bytes saved.
#
# The "the workspace is wasted work" intuition is also wrong for prefill, but not
# for the reason one expects. Dequantizing the whole pool is nearly free and
# amortized -- 0.0145 ms for 8192 rows, ~1% of the attention it feeds -- so there
# is little to win even in principle. (The reuse argument is not the mechanism:
# the fused form loses by 8-10x at T=1, where each row is gathered less than once,
# and the ratio *improves* to 5.5x at T=4096. It is a per-access cost, not a
# repeated-decode cost.)
#
# Revive this if any of these change: (a) the gather becomes bandwidth-bound
# rather than issue-bound (a much larger topk, or an arch where the byte loads
# vectorize), (b) the pool layout is changed so a row's fp8, rope and scale are
# one contiguous run, or (c) it is aimed at decode, where T=1 leaves the
# materialization unamortized and the pool read is the whole cost.
#
# It is kept because it is correct and tested, and because it is the only form
# that needs no dequantized copy at all -- which is what a decode-side or a
# memory-constrained caller would want.
#
# The layout is passed in, so this is "paged fp8 with per-tile ue8m0 block
# scales and a bf16 tail", not "DeepSeek-V4". DSv4's instance (see
# kernels/ops/attention/dsv4/dequant_k_cache.py) is:
#   per token : 448 fp8_e4m3 nope + 64 bf16 rope = 576 contiguous bytes,
#               plus 7 ue8m0 scale bytes (one per 64 nope values) padded to 8
#   per page  : [P x 576 data][P x 8 scale], padded up to a multiple of 576
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _paged_fp8_row_tile(
    fp8_ptr,
    bf16_ptr,
    u8_ptr,
    idx,  # [BLOCK_N] token ids, -1 for empty
    d,
    is_nope,
    rope_col,
    s_lane,
    BLOCK_N: tl.constexpr,
    N_LANES: tl.constexpr,  # D / SCALE_TILE
    NOPE_LANES: tl.constexpr,  # D_NOPE / SCALE_TILE; lanes past this are padding
    SCALE_TILE: tl.constexpr,
    D_NOPE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    """Decode BLOCK_N pool rows into one [BLOCK_N, D] bf16 tile."""
    valid = idx >= 0
    loc = tl.where(valid, idx, 0).to(tl.int64)
    page = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    data_base = page * BYTES_PER_PAGE + in_page * ROW_BYTES
    scale_base = (
        page * BYTES_PER_PAGE + S_OFFSET_BYTES + in_page * SCALE_BYTES_PER_TOKEN
    )

    fp8v = tl.load(
        fp8_ptr + data_base[:, None] + d[None, :],
        mask=valid[:, None] & is_nope[None, :],
        other=0.0,
    ).to(tl.float32)
    # ue8m0: the byte is a biased power-of-two exponent, so the dequant is an
    # exp2 scale, not a multiply by a stored float.
    #
    # Load the scales at their real width (one byte per SCALE_TILE values, so
    # [BLOCK_N, D/SCALE_TILE]) and broadcast, rather than gathering a full
    # [BLOCK_N, D] tile of them. The wide form reads the same 8 bytes per row 64
    # times over and, more importantly, its pipelined buffers push the tile past
    # SM120's 100 KB once BLOCK_H reaches 64.
    sc = tl.load(
        u8_ptr + scale_base[:, None] + s_lane[None, :],
        mask=valid[:, None] & (s_lane < NOPE_LANES)[None, :],
        other=127,
    ).to(tl.float32)
    scale = tl.reshape(
        tl.broadcast_to(tl.exp2(sc - 127.0)[:, :, None], (BLOCK_N, N_LANES, SCALE_TILE)),
        (BLOCK_N, N_LANES * SCALE_TILE),
    )
    nope = fp8v * scale

    rope = tl.load(
        bf16_ptr + (data_base[:, None] + D_NOPE) // 2 + rope_col[None, :],
        mask=valid[:, None] & (is_nope[None, :] == 0),
        other=0.0,
    )
    return tl.where(is_nope[None, :], nope.to(tl.bfloat16), rope), valid


@triton.jit
def _nsa_prefill_paged_fp8_kernel(
    q_ptr,
    fp8_ptr,  # SWA pool bytes viewed as float8_e4m3
    bf16_ptr,  # the same bytes viewed as bfloat16 (the rope tail)
    u8_ptr,  # the same bytes viewed as uint8 (the ue8m0 scales)
    idx_ptr,
    len_ptr,
    x_fp8_ptr,  # compressed pool, same three views; unused when not HAS_EXTRA
    x_bf16_ptr,
    x_u8_ptr,
    x_idx_ptr,
    x_len_ptr,
    o_ptr,
    sink_ptr,
    sm_scale,
    topk,
    x_topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,  # full row width = D_NOPE + D_ROPE (512 for DSv4)
    D_NOPE: tl.constexpr,  # fp8 part; the remainder is the bf16 tail
    SCALE_TILE: tl.constexpr,  # fp8 values per ue8m0 scale byte (64)
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    X_PAGE_SIZE: tl.constexpr,
    X_BYTES_PER_PAGE: tl.constexpr,
    X_S_OFFSET_BYTES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    t = tl.program_id(0)

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    d = tl.arange(0, D)
    is_nope = d < D_NOPE
    # Clamped so the masked-off half of each load still forms a legal address.
    rope_col = tl.where(is_nope, 0, d - D_NOPE)
    s_lane = tl.arange(0, D // SCALE_TILE)

    # V is the whole row (DSv4 has no separate value projection), so one decoded
    # tile feeds both dots and each row is touched exactly once.
    q = tl.load(
        q_ptr + t * H * D + h[:, None] * D + d[None, :],
        mask=hmask[:, None],
        other=0.0,
    )

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)

    n = tl.arange(0, BLOCK_N)

    k_len = tl.load(len_ptr + t)
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        kv, valid = _paged_fp8_row_tile(
            fp8_ptr, bf16_ptr, u8_ptr, idx, d, is_nope, rope_col, s_lane,
            BLOCK_N=BLOCK_N, N_LANES=D // SCALE_TILE,
            NOPE_LANES=D_NOPE // SCALE_TILE, SCALE_TILE=SCALE_TILE,
            D_NOPE=D_NOPE, PAGE_SIZE=PAGE_SIZE, BYTES_PER_PAGE=BYTES_PER_PAGE,
            ROW_BYTES=ROW_BYTES, SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
            S_OFFSET_BYTES=S_OFFSET_BYTES,
        )
        qk = tl.dot(q, tl.trans(kv)) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    if HAS_EXTRA:
        # Online softmax is associative over the concatenation, so the second
        # pool just continues the same running (m, l, acc).
        x_len = tl.load(x_len_ptr + t)
        for k0 in tl.range(0, x_len, BLOCK_N):
            idx = tl.load(
                x_idx_ptr + t * x_topk + k0 + n, mask=(k0 + n) < x_len, other=-1
            )
            kv, valid = _paged_fp8_row_tile(
                x_fp8_ptr, x_bf16_ptr, x_u8_ptr, idx, d, is_nope, rope_col,
                s_lane,
                BLOCK_N=BLOCK_N, N_LANES=D // SCALE_TILE,
                NOPE_LANES=D_NOPE // SCALE_TILE, SCALE_TILE=SCALE_TILE,
                D_NOPE=D_NOPE, PAGE_SIZE=X_PAGE_SIZE,
                BYTES_PER_PAGE=X_BYTES_PER_PAGE, ROW_BYTES=ROW_BYTES,
                SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
                S_OFFSET_BYTES=X_S_OFFSET_BYTES,
            )
            qk = tl.dot(q, tl.trans(kv)) * sm_scale
            qk = tl.where(valid[None, :], qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(qk - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            m_i = m_new

    if HAS_SINK:
        sink = tl.load(sink_ptr + h, mask=hmask, other=-float("inf")).to(tl.float32)
        m_base = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_comb = tl.maximum(m_base, sink)
        rescale = tl.exp(m_base - m_comb)
        l_i = l_i * rescale + tl.exp(sink - m_comb)
        acc = acc * rescale[:, None]

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + t * H * D + h[:, None] * D + d[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


def _paged_fp8_layout(cache, page_size, d_nope, d_rope, scale_tile):
    u8 = cache.view(torch.uint8)
    assert u8.is_contiguous(), "the paged cache must be contiguous"
    row_bytes = d_nope + d_rope * 2
    return (
        u8.view(_FP8_DTYPE).reshape(-1),
        u8.view(torch.bfloat16).reshape(-1),
        u8.reshape(-1),
        u8.shape[-1],  # bytes per page
        row_bytes,
        # The pool pads the per-token scale stride to a power of two (DSv4: 7
        # scales in an 8-byte slot).
        triton.next_power_of_2(d_nope // scale_tile),
        page_size * row_bytes,  # start of the page's scale footer
    )


def sparse_mla_prefill_paged_fp8(
    q,
    quant_k_cache,
    indices,
    sm_scale,
    page_size,
    *,
    extra_cache=None,
    extra_indices=None,
    extra_page_size=None,
    extra_topk_length=None,
    d_nope=448,
    d_rope=64,
    scale_tile=64,
    attn_sink=None,
    topk_length=None,
    out=None,
    config=None,
):
    """Sparse-MLA prefill read straight off one or two paged fp8 KV pools.

    Same contract as ``sparse_mla_prefill`` except that ``indices`` are token ids
    into a pool rather than rows of a pre-gathered block, so no dequantized copy
    is needed anywhere.

    Args:
        q: ``[T, H, d_nope + d_rope]`` bf16.
        quant_k_cache: ``[num_pages, bytes_per_page]``, any dtype, read as bytes.
        indices: ``[T, topk]`` int32 token ids; ``-1`` marks an empty slot.
        page_size: tokens per page of ``quant_k_cache``.
        extra_cache / extra_indices / extra_page_size: an optional second pool
            attended in the same softmax — DSv4's compressed cache, which has its
            own page size (64 at compress_ratio 4, 2 at 128).
        d_nope / d_rope: fp8 and bf16 halves of a row.
        scale_tile: fp8 values covered by one ue8m0 scale byte.
        attn_sink: optional ``[H]`` fp32 per-head sink logit.
    """
    if indices.dim() == 3:
        assert indices.shape[1] == 1
        indices = indices.squeeze(1)
    if extra_indices is not None and extra_indices.dim() == 3:
        assert extra_indices.shape[1] == 1
        extra_indices = extra_indices.squeeze(1)

    T, h, d_qk = q.shape
    D = d_nope + d_rope
    if d_qk != D:
        raise ValueError(
            f"q is {d_qk} wide but a cache row is {D} ({d_nope} + {d_rope})."
        )
    if d_nope % scale_tile:
        raise ValueError(
            f"d_nope {d_nope} must be a multiple of scale_tile {scale_tile}."
        )
    if D & (D - 1):
        raise ValueError(f"row width {D} must be a power of two for the value tile.")

    fp8_p, bf16_p, u8_p, bpp, row_bytes, sbpt, s_off = _paged_fp8_layout(
        quant_k_cache, page_size, d_nope, d_rope, scale_tile
    )
    has_extra = extra_cache is not None
    if has_extra:
        if extra_indices is None or extra_page_size is None:
            raise ValueError(
                "extra_cache needs extra_indices and extra_page_size as well."
            )
        x_fp8, x_bf16, x_u8, x_bpp, _, _, x_s_off = _paged_fp8_layout(
            extra_cache, extra_page_size, d_nope, d_rope, scale_tile
        )
        extra_indices = extra_indices.contiguous()
        if extra_topk_length is None:
            extra_topk_length = _topk_length(extra_indices, extra_indices.shape[-1])
        x_topk = extra_indices.shape[-1]
    else:
        x_fp8, x_bf16, x_u8 = fp8_p, bf16_p, u8_p
        x_bpp, x_s_off, x_topk = bpp, s_off, 1
        extra_indices = indices
        extra_topk_length = None

    indices = indices.contiguous()
    if out is None:
        out = torch.empty(T, h, D, dtype=torch.bfloat16, device=q.device)
    if topk_length is None:
        topk_length = _topk_length(indices, indices.shape[-1])
    if extra_topk_length is None:
        extra_topk_length = topk_length
    if attn_sink is not None:
        if attn_sink.shape != (h,):
            raise ValueError(f"attn_sink must be [{h}]; got {tuple(attn_sink.shape)}.")
        attn_sink = attn_sink.to(torch.float32).contiguous()

    bn, warps, stages = config or _config(q.device, h)
    block_h = _block_h(q.device, h)
    # The decode holds an fp32 staging tile alongside the bf16 one, so a tile the
    # bf16 path fits can miss the smem budget here; step down as it does.
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_paged_fp8_kernel[(T,)](
                q,
                fp8_p,
                bf16_p,
                u8_p,
                indices,
                topk_length,
                x_fp8,
                x_bf16,
                x_u8,
                extra_indices,
                extra_topk_length,
                out,
                attn_sink if attn_sink is not None else topk_length,
                sm_scale,
                indices.shape[-1],
                x_topk,
                H=h,
                BLOCK_H=block_h,
                D=D,
                D_NOPE=d_nope,
                SCALE_TILE=scale_tile,
                PAGE_SIZE=page_size,
                BYTES_PER_PAGE=bpp,
                ROW_BYTES=row_bytes,
                SCALE_BYTES_PER_TOKEN=sbpt,
                S_OFFSET_BYTES=s_off,
                X_PAGE_SIZE=extra_page_size or page_size,
                X_BYTES_PER_PAGE=x_bpp,
                X_S_OFFSET_BYTES=x_s_off,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
                HAS_EXTRA=has_extra,
                HAS_SINK=attn_sink is not None,
            )
            return out
        except triton.runtime.errors.OutOfResources:
            continue
    raise triton.runtime.errors.OutOfResources(
        0, 0, "shared memory: no fallback config fits this device/shape"
    )
