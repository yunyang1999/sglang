# DSv4 sparse-MLA on SM120 — Triton kernel, and what it gains on your fork

For the DeepSeek-V4 team. This answers the question as asked: the kernel is
already integrated into `xutizhou/sglang @ deepseek-base-optimization` and
measured **on your Best Config**, unchanged. Numbers below are from that run,
not from a synthetic harness.

## What it is

One framework-neutral file — module-level imports are `logging`, `torch`,
`triton`, `triton.language`, nothing else:

    python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py

plus ~137 lines of wiring across three existing files. Both entry points are
off by default and only reachable on SM120:

```bash
export SGLANG_DSV4_TRITON_DECODE=1          # this is the one that pays
export SGLANG_DSV4_TRITON_SPARSE_PREFILL=1  # optional; measured 0 end to end
```

Leave `SGLANG_DSV4_TRITON_DECODE_SPLITS=0` (means "choose host-side") and
`SGLANG_DSV4_TRITON_UNION=0`.

```bash
git checkout 2538def09          # or your current deepseek-base-optimization
git apply dsv4-sm120-triton-sparse-mla.patch
```

| file | lines | what |
|---|---|---|
| `.../dsa/triton_sparse_mla_prefill.py` | +4140 | the kernel (new file) |
| `srt/environ.py` | +21 | the switches |
| `srt/layers/attention/deepseek_v4_backend.py` | +99/-9 | decode/prefill dispatch |
| `srt/models/deepseek_v4.py` | +17 | skip head padding on the Triton prefill route |
| `test/.../test_dsa_triton_sparse_mla_prefill.py` | +963 | 33 tests |

The patch is cut **against your `2538def09`, from the tree that produced the
numbers below** — not rebased from ours, which had drifted from yours in
`deepseek_v4_backend.py` and `deepseek_v4.py` and would not have applied.
Verified: `git apply --check` on a pristine `2538def09` checkout is clean with
zero offsets, and all five files byte-compile after applying.

One deliberate difference from the measured tree: the A/B carried path counters
(`dsv4_ab_probe`) to prove which backend each call took. Those are stripped here
— 5 counter blocks and 1 timing region, verified by token scan and re-parse.
They were live during the measurement, so the shipped code is if anything
marginally faster than what was benchmarked.

Our full branch is `dsa-triton-sparse-mla-prefill-dsv4-enable` on
`github.com/yunyang1999/sglang`. It carries some extras this patch leaves out on
purpose — notably a wider head-unpadding refactor that also unpads *decode*.
That is **unmeasured on your tree**, and your fork already has its own
`no_pad_threshold` logic it would interact with, so this patch keeps the
conservative version: unpad on the extend path only, decode stays padded to 64,
exactly as benchmarked.

## End to end, on your fork and your config

Fork at `deepseek-base-optimization` (`2538def09`), config verbatim:

```
--tp 4 --dp 4 --enable-dp-attention --ep 4 --pp-size 1
--moe-runner-backend deep_gemm --chunked-prefill-size 4096
--mem-fraction-static 0.70 --disable-flashinfer-autotune --disable-radix-cache
--skip-server-warmup --cuda-graph-max-bs 16
--dsa-paged-mqa-logits-backend deepgemm --enable-deepseek-v4-fp4-indexer
```

Both arms on one 8x RTX PRO 6000 node, disjoint 4-GPU groups, simultaneous, 3
reps. Arm A = your config as-is. Arm B = identical plus the two switches above.

**LEVEL 1 — end to end** (your protocol: OSL 1024, 64 prompts, rate inf, seed 3)

| ISL | CC | output tput B/A | TTFT B/A | **TPOT B/A** |
|---|---|---|---|---|
| 1024 | 8 | 1.028 | 1.036 | **0.967** |
| 1024 | 32 | 1.036 | 1.032 | **0.961** |
| 1024 | 64 | 1.036 | 1.036 | **0.959** |
| 8192 | 8 | 1.022 | 1.032 | **0.967** |
| 8192 | 32 | 1.013 | 1.014 | **0.971** |
| 8192 | 64 | 1.000 | 1.029 | **0.987** |

**LEVEL 3 — decode alone**: output tput 1.032 / 1.032, TPOT 0.968 / 0.968 at
CC 8 / 64.

**LEVEL 2 — prefill alone**: TTFT B/A 1.008 (ISL 1024), 1.012 (ISL 8192).
**Prefill gains nothing at the request level** — see the caveat below.

**Accuracy**: GSM8K 400q 5-shot, A 0.958 / 0.958 against B 0.960 / 0.955.
Indistinguishable.

Reproduced across two independent runs (jobs 3627369 and 3638276); TPOT B/A
0.960-0.969 and 0.959-0.987.

Routing is confirmed by path counters rather than by the flags: arm A takes
FlashInfer 817 times and this kernel zero, arm B takes this kernel and
FlashInfer **zero**.

## Where the gain comes from — kernel level

Against FlashInfer, which is what your config selects
(`SGLANG_SM120_FLASHMLA_BACKEND=flashinfer`), called **at its own smallest legal
head count** so no padding cost is attributed to it. Both sides read the same
stored fp8 bytes and ue8m0 exponents. RTX PRO 6000, decode under CUDA-graph
replay:

    decode, H=64     B=1    2      4      8     16     32     64    128
                    2.20x  2.20x  2.20x  2.16x  2.22x  2.00x  1.69x 1.29x

`B = 1..16` is **94% of the decode steps your run actually issued** — counted
from `#running-req` in the server log: 2:5683, 8:1387, 1:1223, 16:1176, 4:402.
B=2 alone is 54%.

Holds across the other axes, so it is not one shape getting lucky:

| axis | range | speedup |
|---|---|---|
| candidate width | 128 + {256, 512, 1024} | 1.17 – 2.20x |
| pool size (L2 pressure) | 8k → 64k | 1.29 – 2.20x |
| ragged candidate lists | per-row lengths | 1.29 – 2.00x |
| head count | 8 / 16 / 32 / 64 | 1.17 – 2.50x |
| device | 5080 (84 SM) / 5090 (170) / PRO 6000 (188) | gmean 2.04 / 2.15 / 2.21x |

Prefill is 1.16 – 1.85x at the kernel (gmean 1.38x native fp8, 1.57x bf16).

### The four things it does differently

1. **Head tiling** — one program per (token, head tile) instead of per token. At
   your padded h=64 a monolithic `[64, 512]` fp32 accumulator is 256
   registers/thread over 128 threads against a 255 cap, and spills **9,104 B**.
   A 16-row tile spills 1,320 B at an *identical* mma count. Not a trade: every
   other way of buying occupancy here costs 1.5–2.0x the tensor work.
2. **Native fp8** — the stored fp8 bytes go into the tensor core (QMMA.16832)
   with no dequantised workspace, halving the KV bytes crossing L2.
3. **Split-K decode** with a host-side split policy. At B=1 the unsplit kernel
   leaves one SM of 84 busy; splitting is **4.60x** there.
4. **Merge warp count** — the split-K merge reduces a `[SPLIT_PAD, 512]` bf16
   tile; at 128 threads each takes 8 bytes where the hardware wants a 16-byte
   vector. 64 threads is 1.22x at B=8 and pulled B=8/16 up from 1.83x/1.45x to
   2.16x/2.22x.

## Caveats, stated up front

**Prefill is neutral end to end.** The kernel is 1.16–1.27x at 64 heads but TTFT
does not move (1.008–1.012). Prefill is not attention-bound in this config. If
you only want what pays, set `SGLANG_DSV4_TRITON_DECODE=1` and leave the prefill
switch off.

**The decode ceiling is ~6–8%, and we are at about half of it.** Derived from
your own A/B rather than estimated — both arms differ only in the attention
kernel, so `t_attn = ΔTPOT / (1 - 1/s)`:

| | attention share of a decode step | ceiling on TPOT | needed for −5% |
|---|---|---|---|
| CC=8 (per-rank B≈2) | 6.4% | −6.4% | 4.6x over FlashInfer |
| CC=64 (per-rank B≈16) | 8.0% | −8.0% | 2.7x over FlashInfer |

Attention is 6–8% of a decode step in a 43-layer MoE. An infinitely fast
attention kernel buys −6.4% at CC=8. **−5% end to end is not reachable from this
kernel**, and no amount of tile tuning changes that.

**The advantage narrows with batch and with head count** — 2.20x at B=1 down to
1.29x at B=128. The mechanism is head tiling's own cost: each token's KV is
re-gathered once per head tile (~2x the global sectors). At small batch the
device is idle and that is free; at large batch it is not.

**Two test tolerances were device-dependent, and are now widened.** The bounds
0.06 (`test_matches_bf16_gather`) and 0.01 (`test_split_matches_unsplit`) were
fixed on one part and do not hold on others: an RTX PRO 6000 measures 0.0732 and
0.015625, and an RTX 5090 measures 0.0664 at h=64 — over the old bound as well.
These are absolute-threshold misses, not wrong answers; `cos` is unmoved at
0.999465 / 0.9999969.

Attributed rather than assumed: both values are byte-identical with the merge at
4 warps and at 2, so nothing on this branch moved them, and
`test_matches_bf16_gather` runs at `splits=1` where the merge is never called.
Raised to 0.08 / 0.02 — widest part measured plus ~10%, so a real drift still
trips them — with the measured numbers recorded in the test.

## Things we tried and rejected

Recorded in the module so they are not retried. Nine attempts, all measured
slower or flat, with the profile-based reason each looked good:

| tried | why it looked right | measured |
|---|---|---|
| `BLOCK_N` 64→32 | 3 blocks/SM | 0.72–0.91x |
| `num_warps` 4→8 (split kernel) | zero spill, 2x warps | 0.78x / 0.61x |
| `num_warps` 4→2 (split kernel) | the merge's win | 0.71x / 0.61x |
| `loop_unroll_factor=2` | branch efficiency 11.76% | 0.62x / 0.47x |
| `num_stages` 1/3/4 | latency-bound | ±1% |
| heads per merge program | 512 tiny programs | 0.95–1.00x |
| per-batch (BH, BN, splits) policy | grid fills only 24% at B=1 | 1.013x |
| register-accumulating merge | 86% excessive shared wavefronts, Est. 55.63% | 1.000x |
| one wide 8-byte scale load | 23.2 of 32 bytes per sector, Est. 8.878% | 0.906x |

NCU explains why the occupancy ones all failed: `Block Limit Registers = 2` and
`Block Limit Shared Mem = 2` hold **simultaneously**, so relieving either alone
gains exactly nothing. The kernel is latency-bound on the L1 path (`No Eligible`
73.85%, L1/TEX 50.92% against DRAM 8.02% and 3% of FP32 peak).

The last structural idea — a cluster + DSMEM scheme to gather each token's KV
once for all four head tiles — **cannot be written in Triton**: `num_ctas=4` does
emit `.explicitcluster` on sm_120, but the language exposes no DSMEM primitive
and the PTX carries no `mapa`, so one CTA cannot read another's shared memory.
TMA multicast does not apply either (it needs a descriptor over a contiguous
tile; this gather is index-scattered over a paged pool). It would mean rewriting
the kernel in CUDA C++/PTX, and three independent signals say the payoff is
small: BLOCK_H=32 *is* "half the re-gather" and loses at every batch; L2 hit rate
is 89.36% so the duplicate gather is already absorbed; and the two experiments
above that argued from memory-traffic counts both measured slower.

## Validation

- 33 tests in `test/registered/kernels/ops/attention/test_dsa_triton_sparse_mla_prefill.py`,
  green on **two** sm_120 parts: RTX PRO 6000 (job 3657915, run twice — at merge
  `num_warps` 4 and 2, to separate the merge change from the part) and RTX 5090
  (job 3659011). One skip, which is the FlashInfer cross-check where FlashInfer
  is not importable in the container.
- Accuracy against an fp32 oracle over the **dequantised stored** KV — the values
  the cache actually holds, so quantisation the kernel never chose is not charged
  to it: cos 0.999599, rel-L2 2.83e-2, unchanged by head tiling to six figures.
- Head tiling is bitwise-invariant within a layout family — verified at
  h=16/32/64, through split-K decode at every (B, splits), and on ragged rows
  (job 3659011, "ALL CHECKS PASS"). Across the BLOCK_H=64 boundary, where
  Triton's `warpsPerCTA` flips `[1,4]` → `[4,1]` and the softmax row reduction
  changes order, cos is 1.0000000 with max|d| 0.0010 — under one bf16 ULP. The
  tiling does not move the error at all: at h=64 every tile from 8 to 64
  measures the same max|d| to four decimals.
- Decode must be timed under CUDA-graph replay. FlashInfer's wrapper carries
  ~106 us of host overhead, which eager timing hands you as a ~6x speedup that is
  not real.
- One integration trap worth knowing: **every synthetic pool in the test file is
  contiguous and SGLang's is not.** `deepseek_v4_backend` passes
  `swa_k_cache[:, : swa_window_size * k_cache_total_dim]` reshaped to 4-D, so
  pages sit `stride(0)` bytes apart in a wider buffer. 33 tests and a full grid of
  head-to-heads passed while the server could not start.
  `test_sliced_pool_matches_contiguous` covers it now.

## Files here

    README.md                              this
    dsv4-sm120-triton-sparse-mla.patch     apply to deepseek-base-optimization
    triton_sparse_mla_prefill.py           the kernel on its own
