# DSv4 sparse-MLA on SM120 — results

**Model** DeepSeek-V4 (DSv4-Flash), DSA sparse MLA · `d_qk = d_v = 512` ·
`index_topk 512` + `sliding_window 128` = **640 candidates** over two paged fp8
KV pools (SWA page 256, C4-compressed page 64) · ue8m0 scales · per-head learned
`attn_sink`.

**Baseline** FlashInfer CUTLASS SM120 sparse-MLA — the path the config already
selects (`SGLANG_SM120_FLASHMLA_BACKEND=flashinfer`). Same stored fp8 bytes and
exponents on both sides. Called **at its own smallest legal head count**, so the
framework's head padding is not charged to it. Decode timed under CUDA-graph
replay (its wrapper carries ~106 µs host overhead).

**Ours** the Triton sparse-MLA kernel, via `SGLANG_DSV4_TRITON_DECODE=1`
(+ `SGLANG_DSV4_TRITON_SPARSE_PREFILL=1` where the table says so).

Terms used below, spelled out once: **TTFT** = time to first token (prefill
latency, lower is better) · **TPOT** = time per output token (decode latency,
lower is better) · **input len** / **output len** = tokens in / tokens generated
per request · **concurrency** = simultaneous requests.

---

## 1. Kernel — decode · RTX 5090 (170 SM) · CUDA-graph replay

**How many times faster than the baseline** (higher is better; 1.00x = tied):

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| **8 heads** | 2.37x | 2.49x | 2.49x | 2.29x | 2.00x | 2.00x | 2.50x | 4.61x¹ |
| **64 heads** | 2.00x | 2.00x | 2.20x | 1.72x | 2.11x | 1.65x | 1.76x | 1.16x |

**Geometric mean over the whole batch × heads grid: 2.15x.**
¹ the one cell where no smaller legal head count exists for the baseline, so it
is measured against its padded call.

| axis varied | range tested | speedup vs baseline |
|---|---|---|
| candidate width | 128 + {256, 512} | 1.17 – 2.50x |
| KV pool size (L2 pressure) | 8k → 64k tokens | 1.16 – 2.20x |
| ragged rows (each row a different real length) | batch 8 / 32 / 128 | 1.18 – 2.00x |
| head count | 8 / 16 / 32 / 64 | 1.16 – 2.50x |
| GPU | 5080 (84 SM) / 5090 (170) / PRO 6000 (188) | gmean **2.04 / 2.15 / 2.21x** |

**On RTX PRO 6000** (the GPU the end-to-end ran on), 64 heads, batch 1 → 128:
2.20 / 2.20 / 2.20 / 2.16 / 2.22 / 2.00 / 1.69 / 1.29x. **Batch 1–16 is 94% of
the decode steps the end-to-end run actually issued** (counted from
`#running-req` in the server log: batch 2 → 5683 steps, 8 → 1387, 1 → 1223,
16 → 1176, 4 → 402; batch 2 alone is 54%).

## 2. Kernel — prefill · RTX 5090 · eager

**How many times faster than the baseline:**

| tokens in the chunk | 8 heads | 16 heads | 32 heads | 64 heads |
|---|---|---|---|---|
| 512 | 1.78x | 1.67x | 1.34x | 1.20x |
| 1024 | 1.74x | 1.59x | 1.30x | 1.37x |
| 2048 | 1.82x | 1.63x | 1.43x | 1.28x |
| 4096 | 2.05x | 1.77x | 1.36x | 1.20x |
| 8192 | 1.98x | 1.78x | 1.33x | 1.18x |

Flat in chunk size; narrows as heads widen. At DSv4's padded **64 heads:
1.18–1.37x**.

## 3. End to end · 8× RTX PRO 6000 · SGLang `xutizhou/deepseek-base-optimization` @ `2538def09`

Config **verbatim, unchanged**:

```
--tp 4 --dp 4 --enable-dp-attention --ep 4 --pp-size 1 --moe-runner-backend deep_gemm
--chunked-prefill-size 4096 --mem-fraction-static 0.70 --disable-flashinfer-autotune
--disable-radix-cache --skip-server-warmup --cuda-graph-max-bs 16
--dsa-paged-mqa-logits-backend deepgemm --enable-deepseek-v4-fp4-indexer
```

**Method.** Two servers on one node at the same time, on separate halves of the
GPUs: one running the config as-is (**baseline**), one running the identical
config plus the two switches (**ours**). 3 repetitions each. Then the whole thing
**run a second time with the two servers swapped onto the other half of the
node**, to cancel any difference between the two GPU groups. Both runs are shown
— they did not agree as closely as expected, which is itself a result.

Every number below is **ours ÷ baseline**.

### 3a. Serving — output len 1024, 64 prompts, unlimited request rate, seed 3

| input len | concurrency | throughput (higher better) | TTFT (lower better) | TPOT run 1 | TPOT run 2 | **TPOT mean (lower better)** |
|---|---|---|---|---|---|---|
| 1024 | 8 | 1.032 | 0.976 | 0.965 | 0.975 | **0.970** |
| 1024 | 32 | 1.044 | 0.989 | 0.945 | 0.968 | **0.956** |
| 1024 | 64 | 1.046 | 0.985 | 0.944 | 0.962 | **0.953** |
| 8192 | 8 | 1.034 | 0.991 | 0.952 | 0.975 | **0.964** |
| 8192 | 32 | 1.030 | 0.976 | 0.949 | 0.982 | **0.966** |
| 8192 | 64 | 1.034 | 0.970 | 0.937 | 0.998 | **0.968** |

**TPOT 3.0% to 4.7% better, at every one of the six points.** Throughput
+3.0% to +4.6%. TTFT is not a target here (the prefill switch is off) and lands
slightly under 1.0 at every point.

> **How much to trust a single row.** Swapping the two servers onto the other
> half of the node moves TPOT by **+2.8 percentage points on average** (run 2
> minus run 1, range +1.0 to +6.1). That is the resolution of this experiment.
> Read the mean column; treat anything under ~3 pp as unresolved. We learned
> this the hard way — a kernel change looked 0.6 pp better on one placement and
> turned out to be nothing once both were run.

### 3b. Decode in isolation

| concurrency | throughput run 1 | throughput run 2 | TPOT run 1 | TPOT run 2 |
|---|---|---|---|---|
| 8 | 1.035 | 1.047 | 0.961 | 0.949 |
| 64 | 1.043 | 1.049 | 0.957 | 0.950 |

### 3c. Prefill in isolation — output len 1, concurrency 1, so no queueing

| input len | TTFT run 1 | TTFT run 2 |
|---|---|---|
| 1024 | 0.972 | 0.997 |
| 8192 | 0.963 | 1.010 |

**Neutral** — lands on both sides of 1.00 across the two runs, and the spread
between placements is larger than the effect.

Why, measured rather than argued: CUDA-event timing inside the running server
shows the kernel *is* **1.565x** there (baseline 484.5 µs kernel + 105.1 µs page
transcode = 589.6 µs per call; ours 376.7 µs) — the bench speedup is fully
realised. But attention is only **2.1% of TTFT at input len 1024, and 5.5% at
8192**, so 1.57x on it is worth at most 1.4% of TTFT. **Recommend leaving the
prefill switch off.**

---

## 4. Accuracy

### 4a. Model accuracy — GSM8K, 400 questions, 5-shot, temperature 0

Same weights, same sampling, same prompts; only the attention kernel differs.

| | run 1 rep 1 | run 1 rep 2 | run 2 rep 1 | run 2 rep 2 | mean | invalid answers |
|---|---|---|---|---|---|---|
| **baseline** (FlashInfer) | 0.963 | 0.963 | 0.953 | 0.965 | 0.961 | 0.0 |
| **ours** (Triton kernel) | 0.960 | 0.963 | 0.958 | 0.968 | 0.962 | 0.0 |

**Indistinguishable** — our spread (0.958–0.968) sits inside the baseline's
(0.953–0.965).

### 4b. Numerical accuracy — against an fp32 reference

The reference is computed from the **dequantised values the KV cache actually
holds**, so quantisation error the kernel never chose is not charged to it.

| metric | value |
|---|---|
| cosine similarity | **0.999599** |
| relative L2 error | **2.83e-2** |
| change caused by head tiling | none, to 6 significant figures |

### 4c. Invariance — what is bit-identical and what is not

| property | result |
|---|---|
| head tiling within one layout family (16/32/64 heads, every batch × split count, ragged rows) | **bit-identical** |
| merge warp count 1 vs 2, every batch × split count, two GPUs | **bit-identical** |
| `splits=1` vs the unsplit kernel | **bit-identical** |
| across the 64-row tile boundary (Triton's `warpsPerCTA` flips `[1,4]`→`[4,1]`, so the softmax reduction reorders) | cosine 1.0000000, max abs diff 0.0010 — **under one bf16 ULP** |
| split-K with more than one split | algebraically exact (log-domain merge), **not** bit-identical — the softmax is reassociated |

### 4d. Test suite

**36 of 36 pass** on RTX PRO 6000, RTX 5080 and RTX 5090 — under the default
merge policy *and* with the merge warp count forced to 1, 2 and 4. One skip: a
cross-check against FlashInfer, which is not importable in that container.

---

## 5. Hardware utilisation · RTX PRO 6000 · denominators measured on this GPU

Not quoted from a spec sheet — measured here with a large GEMM per dtype and a
256 MB copy: **fp8 749.9 TFLOP/s**, **bf16 392.5 TFLOP/s**, **1460 GB/s**.

| | achieved | % of GEMM peak | % of copy bandwidth |
|---|---|---|---|
| decode, batch 1 | 6.8 TFLOP/s | 0.90% | 2.0% |
| decode, batch 16 | 57.2 TFLOP/s | 7.63% | 17.2% |
| decode, batch 128 | 80.7 TFLOP/s | 10.76% | 24.3% |
| prefill, 4096 tokens | 109.1 TFLOP/s | 27.81% (of bf16) | — |

Arithmetic intensity is **227.6 FLOP/byte** against this GPU's ridge point of
**513.5**, so the problem is **memory-bound** — a low percentage of compute peak
is the expected answer, not a defect. It is fixed by DSv4's `d=512, topk=640`
and no kernel change moves it. At batch 1–16 neither limit binds: the launch is
40 thread blocks against 376 slots, **11% of the GPU** — a parallelism limit.

**Ceiling.** Attention is **5.5–8.6% of one decode step**, derived from the A/B
itself (`share = (1 − TPOT ratio) / (1 − 1/kernel speedup)`), per config:

| input len | 1024 | 1024 | 1024 | 8192 | 8192 | 8192 |
|---|---|---|---|---|---|---|
| concurrency | 8 | 32 | 64 | 8 | 32 | 64 |
| attention share of a decode step | 5.5% | 8.1% | 8.6% | 6.7% | 6.4% | 5.9% |

An infinitely fast attention kernel would buy **−8.6% TPOT at best**. We measured
−3.0 to −4.7%, i.e. **roughly half of what is there to take**.

---

*Method, kernel design, caveats and the rejected experiments:
`README.md` / `docs/DSV4_FORK_INTEGRATION.md`.*
