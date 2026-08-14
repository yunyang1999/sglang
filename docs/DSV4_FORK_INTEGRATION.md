# DSv4 sparse-MLA on SM120 — Triton kernel, and what it gains on your fork

For the DeepSeek-V4 team. This answers the question as asked: the kernel is
already integrated into `xutizhou/sglang @ deepseek-base-optimization` and
measured **on your Best Config**, unchanged. Numbers below are from that run,
not from a synthetic harness.

## What it is

One framework-neutral file — module-level imports are `logging`, `torch`,
`triton`, `triton.language`, nothing else:

    python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py

plus ~156 lines of wiring across three existing files. Both entry points are
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
| `.../dsa/triton_sparse_mla_prefill.py` | +4201 | the kernel (new file) |
| `srt/environ.py` | +21 | the switches |
| `srt/layers/attention/deepseek_v4_backend.py` | +118/-9 | decode/prefill dispatch + capability guard |
| `srt/models/deepseek_v4.py` | +17 | skip head padding on the Triton prefill route |
| `test/.../test_dsa_triton_sparse_mla_prefill.py` | +1039 | 36 tests |

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

Both arms on one 8x RTX PRO 6000 node, disjoint 4-GPU groups, 3 reps. Arm A =
your config as-is. Arm B = identical plus the two switches above.

Run **twice with the arms swapped between the two 4-GPU groups** (jobs 3661108,
3661109), and both placements are reported, because the swap turned out to
matter more than we expected — see the note under the table.

**LEVEL 1 — end to end** (your protocol: OSL 1024, 64 prompts, rate inf, seed 3)

| ISL | CC | tput B/A | TTFT B/A | TPOT R1 | TPOT R2 | **TPOT mean** |
|---|---|---|---|---|---|---|
| 1024 | 8 | 1.031 | 1.002 | 0.966 | 0.975 | **0.970** |
| 1024 | 32 | 1.034 | 1.024 | 0.959 | 0.968 | **0.964** |
| 1024 | 64 | 1.039 | 1.015 | 0.953 | 0.962 | **0.958** |
| 8192 | 8 | 1.026 | 1.016 | 0.962 | 0.975 | **0.968** |
| 8192 | 32 | 1.013 | 1.019 | 0.964 | 0.982 | **0.973** |
| 8192 | 64 | 1.001 | 1.024 | 0.972 | 0.998 | **0.985** |

**TPOT 0.958–0.985 — 1.5 to 4.2% better, at every one of the six points.**

**How much to trust a difference this size.** The placement swap alone moves
TPOT by **+1.4 pp on average** (R2 − R1, range +0.9 to +2.6). That is the
resolution of this experiment, and it is worth stating because it is *larger
than some changes we were tempted to claim*: an earlier kernel revision
measured at one placement looked +0.6 pp better, and running the other
placement showed the real effect was +0.1 pp [−0.2, +0.3] — i.e. nothing. Read
the mean column, not a single row, and treat anything under ~1.5 pp here as
unresolved.

**LEVEL 3 — decode alone**: tput B/A 1.034 / 1.034 (R1) and 1.047 / 1.049 (R2);
TPOT 0.966 / 0.966 and 0.949 / 0.950, at CC 8 / 64.

**LEVEL 2 — prefill alone** (CC 1, so no queueing): TTFT B/A 0.983 / 0.998 (R1)
and 0.997 / 1.010 (R2), for ISL 1024 / 8192. Straddles 1.0 in both placements:
**prefill is neutral at the request level, neither better nor worse.** Why, with
the measurement, is under "Caveats".

**Accuracy**: GSM8K 400q 5-shot, 2 reps x 2 placements. A: 0.963, 0.963, 0.953,
0.965. B: 0.960, 0.963, 0.958, 0.968. Indistinguishable — B's spread sits inside
A's.

Routing is confirmed by path counters rather than by the flags: arm A takes
FlashInfer and this kernel zero times, arm B takes this kernel and FlashInfer
**zero**.

## How the kernel is built

What it has to do to be *correct* for DSv4, before any question of speed. This
is the part to read if you are reviewing or maintaining it.

**One fused pass over two KV pools with different page sizes.** DSv4-Flash
selects `index_topk=512` from the compressed cache and `sliding_window=128` from
the SWA cache — 640 candidates per query, living in two pools paged at 64 and
256 respectively. The kernel takes both (`extra_cache`/`extra_page_size`) and
walks them in a single softmax, rather than running twice and merging: one
online-softmax running max, one accumulator, one pass. Per-row candidate counts
are passed in (`topk_length`, `extra_topk_length`) so short rows cost nothing.

**It reads the stored fp8 bytes directly — no dequantised workspace.** A row is
`448` fp8 bytes then `64` bf16 rope values (576 B), and the per-token ue8m0
exponents live in a footer at the end of each page: the interior is
`[P x row_bytes data][P x scale_bytes footer]`, with 7 scale bytes padded to 8.
Indices are absolute token ids — there is no page table indirection. The fp8
tile goes into the tensor core as an mma operand in either orientation
(QMMA.16832) and is never materialised as bf16; the scale is applied as one byte
per row, not broadcast into a `[BLOCK_N, 512]` tile.

**Non-contiguous pools are supported, and SGLang needs that.** The backend hands
over `swa_k_cache[:, : swa_window_size * k_cache_total_dim]` reshaped to 4-D — a
slice along dim 1, so consecutive pages sit `stride(0)` bytes apart inside a
*wider* parent buffer. The kernel takes the stride rather than assuming
contiguity. Getting this wrong is not a wrong answer, it is a crash or a silent
whole-pool copy; see the trap note under Validation.

**The learned per-head sink joins the softmax denominator, exactly once.**
`attn_sink` is a raw natural-log logit with no value row, folded in the way
`_apply_attn_sink` does it — `logaddexp(lse, sink)`. Under split-K it is
deliberately *not* visible to the split kernel: each split would otherwise add
it again. It enters at the merge, and it enters the global max first, so
`exp(sink - gm)` cannot overflow when every real logit sits far below it.

**Split-K is exact, in the log domain.** Splits produce partial `(out, m, l)`;
the merge takes the global max across splits, rescales each by `exp(m_i - gm)`,
sums, and divides once. Algebraically identical to the unsplit kernel, not
bitwise — the softmax is reassociated and partials round to `partial_dtype`.
`splits=1` reproduces the unsplit kernel bit for bit, which is what
`SGLANG_DSV4_TRITON_DECODE_SPLITS=1` gives you if you ever need to bisect.

**Head tiling is bitwise-invariant within a layout family.** Splitting the grid
by head block changes which program computes a row, not the arithmetic — so
tiles 8/16/32 are bit-identical to each other. The exception is documented
rather than hidden: at BLOCK_H=64 Triton's `warpsPerCTA` flips `[1,4]` → `[4,1]`
and the softmax row reduction becomes intra-warp instead of a cross-warp tree,
so results differ in the last bits (cos 1.0000000, max|d| 0.0010, under one bf16
ULP). Comparisons are only meaningful within a family, and the tests pin
`BLOCK_N` when they compare tiles for exactly this reason.

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

These ratios pre-date the SPLIT_PAD merge rule, which is worth a further 1.050x
on the decode call, so they understate the shipped kernel by about that much.
Left as measured rather than scaled: a number we ran is worth more than a number
we multiplied, and erring low is the right direction for a figure you may quote.

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
4. **Merge warp count, chosen from `SPLIT_PAD`** — the split-K merge reduces a
   `[SPLIT_PAD, 512]` bf16 tile along axis 0, across threads. Going 4 → 2 warps
   was 1.22x at B=8 and pulled B=8/16 up from 1.83x/1.45x to 2.16x/2.22x. Going
   further to 1 warp is a further **1.050x on the whole decode call** (PRO 6000;
   1.028x on a 5080) — but only where the tile is tall: 1.29x at SPLIT_PAD 16,
   1.02x at 8, and a *loss* at 4 and 2, where there is nothing to reduce and the
   threads are wanted for the 512-wide store. So the count is a function of
   SPLIT_PAD, not a constant, and it is never worse than the previous rule at
   any shape measured on either part.

   Do not go looking for that 1.050x in the TPOT table: we re-ran the full e2e
   for it across both placements and it came out at **+0.1 pp [−0.2, +0.3]** —
   unresolvable, because attention is a single-digit share of a decode step and
   the placement swap alone is worth +1.4 pp. It is a real kernel-level gain and
   it is free, which is why it ships; it is not an end-to-end claim.

## Caveats, stated up front

**Prefill is neutral end to end, and we measured why.** The kernel speedup is
fully realised inside your server — CUDA-event timing around the dispatch, both
arms, real requests:

| | per prefill attention call |
|---|---|
| arm A | 484.5 us FlashInfer kernel + 105.1 us page transcode (256→64) = **589.6 us** |
| arm B | **376.7 us** |
| | **1.565x**, against 1.57x on the bench — nothing is lost in integration |

It does not show up in TTFT because of what it is a fraction of:

| ISL | TTFT | attention side, per request | **share of TTFT** |
|---|---|---|---|
| 1024 | 304 ms | 6.3 ms | **2.1%** |
| 8192 | 1306 ms | 72.1 ms | **5.5%** |

1.57x on 2.1–5.5% is 0.26–1.4% of TTFT at best, which is under the run-to-run
spread — and that is exactly what the measurement shows. **If you only want what
pays, set `SGLANG_DSV4_TRITON_DECODE=1` and leave the prefill switch off.**

Two things this rules out, both of which we suspected first and had to drop:
the extra bf16 workspace arm B materialises is **not** the reason (measured 4.7%
of the workspace+attention pair at your chunk size of 4096, and that estimate
over-counts the gather); and TTFT is **not** regressing — the LEVEL 1 TTFT
column reads above 1.0 in places, but at ISL 8192 / CC 64 that TTFT is 21 s for
a prefill that computes in well under a second, so LEVEL 1 TTFT is dominated by
queueing under `rate inf` and is not a prefill measurement at all. LEVEL 2 is.

**The decode ceiling is 2.4–7.5%, and we are at most of it.** Derived from your
own A/B rather than estimated: both arms differ only in the attention kernel, so
the whole TPOT delta is attention, and `attention share = (1 - TPOT_B/A) /
(1 - 1/s)` where `s` is the kernel speedup at that config's per-rank batch.

| ISL | CC | per-rank B | TPOT B/A | s | attention share | ceiling on TPOT | needed for −5% |
|---|---|---|---|---|---|---|---|
| 1024 | 8 | 2 | 0.967 | 2.20 | 6.1% | −6.1% | 5.8x |
| 1024 | 32 | 8 | 0.961 | 2.16 | 7.3% | −7.3% | 3.2x |
| 1024 | 64 | 16 | 0.959 | 2.22 | 7.5% | −7.5% | 3.0x |
| 8192 | 8 | 2 | 0.967 | 2.20 | 6.1% | −6.1% | 5.8x |
| 8192 | 32 | 8 | 0.971 | 2.16 | 5.4% | −5.4% | 13.5x |
| 8192 | 64 | 16 | 0.987 | 2.22 | **2.4%** | −2.4% | impossible |

Attention is a **single-digit percentage of a decode step** in a 43-layer MoE —
the other 92–98% is MoE GEMMs, the indexer, dispatch/combine and the rest of the
block. An *infinitely fast* attention kernel buys −7.5% at the best of these
points and −2.4% at the worst. **−5% end to end is not reachable from an
attention kernel here**, and no amount of tile tuning changes that.

This also explains the shape of the e2e table: the config where attention
matters least (ISL 8192, CC 64, share 2.4%) is exactly the one where TPOT barely
moved (0.987), and the configs where it matters most are where we gained most.
The kernel's 2.2x is being applied to a small slice, and 2.2x on 7% is 3.8%.

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

## How much of the machine is it using

"2.2x over FlashInfer" says we beat the incumbent, not that the part is used up.
Denominators measured on the same GPU rather than taken from a spec sheet — a
large square GEMM per dtype, and a 256 MB copy — because marketing peaks
generally include 2:4 sparsity this kernel cannot use.

RTX PRO 6000 Blackwell measures **749.9 TFLOP/s fp8**, **392.5 TFLOP/s bf16**,
**1460 GB/s** copy.

| | achieved | % of measured GEMM peak | % of copy bandwidth |
|---|---|---|---|
| decode B=1 | 6.8 TFLOP/s | 0.90% | 2.0% |
| decode B=16 | 57.2 TFLOP/s | 7.63% | 17.2% |
| decode B=128 | 80.7 TFLOP/s | 10.76% | 24.3% |
| prefill T=4096 | 109.1 TFLOP/s | 27.81% (bf16) | — |

**The low FLOP fractions are the right answer, not a defect.** Arithmetic
intensity is **227.6 FLOP/byte** against this part's ridge point of **513.5**, so
the shape is memory-bound — and the intensity is fixed by DSv4's `d=512,
topk=640`. No kernel change moves it.

**But at the batch sizes that matter, neither roofline binds.** B=1–16 sits at
2–17% of bandwidth and 0.9–7.6% of compute: the kernel is waiting, which is what
NCU says too (`No Eligible` 73.85%, L1/TEX 50.92% against DRAM 8.02%). The
reason is parallelism, not efficiency — at B=1 the launch is 4 head tiles x 10
splits = **40 blocks against 376 block slots, 11% of the device**. The KV a B=1
step must move is 369 KB, which is 0.25 us of bandwidth against 12.4 us measured.
That 49x is a floor, not a target: 369 KB of work cannot fill 188 SMs however
the kernel is written.

Two things follow, and they are worth stating plainly. Prefill, at 28% of bf16
GEMM peak, is the part that is in good absolute shape — and it is the part worth
nothing end to end. And the nine rejected experiments below are consistent with
this picture rather than surprising: if the bottleneck is neither compute nor
bandwidth but latency at low occupancy, then tile, occupancy and traffic-saving
changes should not win, and none of them did.

## How general is this — what is tuned and what is not

Worth being explicit, because the answer differs by layer.

**Structural, no table, follows any device or shape.** The native fp8 gather
(no dequantised workspace); split-K, whose split count `auto_splits` derives from
the runtime SM count; and the merge warp count, which is keyed on `SPLIT_PAD` —
a property of the *shape*, not of the part — so it re-derives itself on hardware
we have never run. Split-K alone is 4.60x at B=1.

**Pinned to (arch, head count).** Head tiling — the single largest win, 1.72–2.13x
on decode — fires only on entries in `_PINNED_NATIVE_HEAD_TILE`, currently
`{(12,0): {32:16, 64:16}}`. Anything else falls back to the untiled kernel:
correct, and about half the speed. Head-count coverage is less narrow than it
looks, since at H≤16 the tile would equal H and tiling is a no-op by
construction, so {32, 64} is the complete set where it can help. The **arch** key
is the real limit: one entry.

**Two cliffs found while writing this, both now fixed.**

* *sm_121 crashed instead of falling back.* `is_sm120_supported()` is true for
  the whole 12.x major, but only sm_120 has the e4m3 mma; the kernel refuses
  sm_121 (it upcasts, which would make this path slower than the one it
  replaces) by raising, and nothing caught it — so on an sm_121 part the decode
  switch took down the server on its first decode step. The backend now asks
  `_has_fp8_mma()` once at import and logs a warning and stays on FlashInfer.
  Your cluster has an `rtx-pro-6000-blackwell-workstation-edition` partition, so
  this was reachable, not hypothetical.
* *The untiled fallback was unbounded in H.* Falling back to "one program holds
  every head" is the right default across devices — it reproduces the kernel the
  table was measured against — and the wrong one across head counts, because the
  fp32 accumulator is `[BLOCK_H, d_v]` and so costs registers linearly in H. At
  H=128 that is 256 KB for the block, which nothing had ever run. Bounded at
  `_MONO_CAP = 64`, the widest tile known to compile and run (it was the shipping
  configuration before head tiling). Every head count ever measured here has
  mono ≤ 64, so this changes nothing that has been run and only bounds the
  regime that had no answer.

**Measured coverage**, so the generality claim is not just structural:

| axis | range | speedup |
|---|---|---|
| head count | 8 / 16 / 32 / 64 | 1.17 – 2.50x |
| candidate width | 128 + {256, 512, 1024} | 1.17 – 2.20x |
| pool size (L2 pressure) | 8k → 64k | 1.29 – 2.20x |
| ragged candidate lists | per-row lengths | 1.29 – 2.00x |
| batch | 1 → 128 | 2.20 → 1.29x |
| device | 5080 / 5090 / PRO 6000 | gmean 2.04 / 2.15 / 2.21x |

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

- 36 tests in `test/registered/kernels/ops/attention/test_dsa_triton_sparse_mla_prefill.py`,
  green on **two** sm_120 parts — RTX PRO 6000 and RTX 5080 (jobs 3660704 /
  3660705) — under the default merge policy *and* under merge `num_warps` forced
  to 1, 2 and 4, so the warp count is separated from the part. Also green on an
  RTX 5090 (3659011). One skip, the FlashInfer cross-check, where FlashInfer is
  not importable in the container.
- The merge's warp count does not move the answer: w=1 is **bitwise equal** to
  w=2 at every (B, splits) measured on both parts, despite being a shorter
  cross-thread reduction.
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
