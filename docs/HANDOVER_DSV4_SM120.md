# DeepSeek-V4 sparse-MLA on SM120 — what to turn on, what it is worth, what it is not

Handover for the DSv4 SGLang work. Branch
`dsa-triton-sparse-mla-prefill-dsv4-enable` on `yunyang1999/sglang`.

## Turning it on

Two independent switches, both off by default, both on the SM120 path only:

```bash
export SGLANG_DSV4_TRITON_DECODE=1          # the one that pays
export SGLANG_DSV4_TRITON_SPARSE_PREFILL=1  # measured 0 at the request level
```

`SGLANG_DSV4_TRITON_DECODE_SPLITS=0` means "choose host-side"; leave it there.
`SGLANG_DSV4_TRITON_UNION` stays 0 — it only pays above ~0.75 neighbour
retention, which DSv4 does not reach.

Everything is a single framework-neutral file:
`python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py`
(module-level imports are `logging`, `torch`, `triton`, `triton.language`).

## What it is worth, at the kernel

Against FlashInfer — SGLang's real SM120 baseline, called **at its own smallest
legal head count**, both sides reading the same stored fp8 bytes and ue8m0
exponents. RTX PRO 6000, 188 SMs, decode under CUDA-graph replay.

    decode, H=64            B=1    2      4      8     16     32     64    128
                           2.20x 2.20x  2.20x  2.16x  2.22x  2.00x  1.69x 1.29x

    prefill, vs FlashInfer at its own head count, gmean over T=512..8192
      H=16 / H=32 / H=64 ......... 1.38x (native fp8), 1.57x (bf16 gather)

`B = 1..16` covers **94% of the decode steps** the e2e run actually issued
(counted from `#running-req` in the server log: 2:5683, 8:1387, 1:1223, 16:1176,
4:402, 5:338, 6:337 — B=2 alone is 54%).

Holds across the other axes too, so this is not one shape getting lucky:

| axis | range | speedup |
|---|---|---|
| candidate width | 128+{256,512,1024} | 1.14 – 2.09x |
| pool size (L2 pressure) | 8k → 64k | 1.22 – 3.42x |
| ragged candidate lists | per-row lengths | 1.26 – 2.00x |
| head count | 8 / 16 / 32 / 64 | 1.17 – 2.22x |

## What it is worth, end to end

Both arms on one 8x RTX PRO 6000 node, the colleague's config verbatim
(TP4 DP4 EP4, `--enable-dp-attention` so `attn_tp_size=1` and all 64 heads are
real, page_size 256, `--cuda-graph-max-bs 16`):

* **TPOT down 3.2–4.1% in every configuration**, never up, across two
  independent runs (jobs 3627369 and 3638276).
* e2e output throughput 1.00–1.04x.
* **prefill alone: 1.008–1.012x TTFT, i.e. nothing.**
* GSM8K 400q 5-shot: A 0.958 / 0.958, B 0.960 / 0.955 — indistinguishable.

The merge fix below is worth 5.5% on the attention kernel and **does not show up
end to end**: 5.5% of a 7% slice is 0.4% of TPOT, under the run-to-run spread.
The two runs bracket it — 0.960-0.969 before, 0.959-0.987 after. That is the
ceiling doing its work, not a broken measurement, and it is the reason the
section below exists.

### The ceiling, so nobody is disappointed

Derived from the A/B itself rather than guessed. Both arms differ only in the
attention kernel, so `t_attn = ΔTPOT / (1 - 1/s)`:

| | attention share of a decode step | ceiling on TPOT | needed for −5% |
|---|---|---|---|
| CC=8 (per-rank B≈2) | 6.4% | −6.4% | 4.6x over FlashInfer |
| CC=64 (per-rank B≈16) | 8.0% | −8.0% | 2.7x over FlashInfer |

Attention is 6–8% of a decode step in a 43-layer MoE. **An infinitely fast
attention kernel buys −6.4% at CC=8.** We are at 2.2x and capture about half of
what is there. −5% end to end is not reachable from this kernel, and no tile
tuning changes that; it would take a different target.

## Where the last kernel gain came from, and where the walls are

The most recent change is one constant: the split-K **merge** ran 128 threads
over a `[SPLIT_PAD, 512]` bf16 tile, which is a 4-column, 8-byte access per
thread where the hardware wants 16. NCU had it at L1/TEX 61.11% with DRAM at
36.76%. 64 threads → 1.22x at B=8, 1.16x at B=16, never worse anywhere, and it
is what pulled B=8 and B=16 up from 1.83x/1.45x to 2.16x/2.22x.

Seven things were tried and **measured slower** — all recorded in the module so
they are not retried:

| tried | why it looked good | measured |
|---|---|---|
| `BLOCK_N` 64→32 | 3 blocks/SM | 0.72–0.91x |
| `num_warps` 4→8 (split) | zero spill, 2x warps | 0.78x / 0.61x |
| `num_warps` 4→2 (split) | the merge's win | 0.71x / 0.61x |
| `loop_unroll_factor=2` | branch efficiency 11.76% | 0.62x / 0.47x |
| `num_stages` 1/3/4 | latency-bound | ±1% |
| heads per merge program | 512 tiny programs | 0.95–1.00x |
| per-batch (BH, BN, splits) policy | grid fills only 24% at B=1 | 1.013x |

NCU explains why the occupancy ones all failed: at BLOCK_H=16 the report carries
`Block Limit Registers = 2` **and** `Block Limit Shared Mem = 2` at once, so
relieving either alone gains exactly nothing. The kernel is latency-bound on the
L1 path (`No Eligible` 73.85%, L1/TEX 50.92% against DRAM 8.02% and 3% of FP32
peak), not occupancy-bound.

Two levers remain, neither cheap:

* **KV-gather coalescing** — 22 of every 32 bytes per sector are used, 19%
  excessive sectors, worth an estimated 13–16%. The signature is non-32B-aligned
  row starts, so the fix is in the **KV cache layout**, i.e. SGLang's side, not
  this kernel's.
* **Redundant gather across head tiles** — head tiling re-gathers each token's
  KV once per head tile; NCU shows ~2x the global sectors of the untiled tile.
  This is the mechanism behind the advantage narrowing at large batch (2.20x at
  B=1 → 1.29x at B=128). A cluster + DSMEM decomposition is the lever; SM120 has
  no CLC, so it would be a hand-rolled software scheme.

## Validation

* 33 tests in
  `test/registered/kernels/ops/attention/test_dsa_triton_sparse_mla_prefill.py`,
  all passing on sm_120.
* Accuracy against an fp32 oracle over the **dequantised stored** KV — the
  values the cache actually holds, so quantisation the kernel never chose is not
  charged to it: cos 0.999599, rel-L2 2.83e-2, identical to the pre-change tile
  to six figures.
* One trap worth knowing: **every synthetic pool in the test file is
  contiguous, and SGLang's is not.** `deepseek_v4_backend` passes
  `swa_k_cache[:, : swa_window_size * k_cache_total_dim]` reshaped to 4-D, so
  pages sit `stride(0)` bytes apart in a wider buffer. 33 tests and a full grid
  of head-to-heads passed while the server could not start.
  `test_sliced_pool_matches_contiguous` covers it now.
* Decode must be timed under CUDA-graph replay. FlashInfer's wrapper carries
  ~106 us of host overhead, which eager timing hands you as a ~6x speedup that
  is not real.

## Reproducing any of it

Probes are in `/home/scratch.yuny_wwfo/nsa_kernel/probe`, indexed in
`README_DSV4_SM120.md`. The ones behind the numbers above:

    probe_h2h_native_grid.py     the FlashInfer head-to-head, all axes
    probe_merge_heads.py         the merge warp/head sweep
    probe_native_tile_sweep.py   the split-kernel tile and warp sweep
    probe_batch_adaptive.py      per-batch (BH, BN, splits) optimum
    probe_native_head_tile.py    head-tiling equivalence
    aot_sm120_*.py               host-side sm_120 resource reads, no GPU needed
