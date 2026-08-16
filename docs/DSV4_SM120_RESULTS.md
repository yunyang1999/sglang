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

**Under test** `SGLANG_DSV4_TRITON_DECODE=1` (+ `_SPARSE_PREFILL=1` where noted).

---

## 1. Kernel — decode · RTX 5090 (170 SM) · CUDA-graph replay · speedup vs baseline

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| **H=8** | 2.37x | 2.49x | 2.49x | 2.29x | 2.00x | 2.00x | 2.50x | 4.61x¹ |
| **H=64** | 2.00x | 2.00x | 2.20x | 1.72x | 2.11x | 1.65x | 1.76x | 1.16x |

**gmean over the B × H grid: 2.15x**  ¹ no smaller legal head count exists here, so vs padded call.

| axis | range | speedup |
|---|---|---|
| candidate width | 128 + {256, 512} | 1.17 – 2.50x |
| pool size (L2 pressure) | 8k → 64k | 1.16 – 2.20x |
| ragged rows (mixed real lengths) | B = 8 / 32 / 128 | 1.18 – 2.00x |
| head count | 8 / 16 / 32 / 64 | 1.16 – 2.50x |
| device | 5080 (84 SM) / 5090 (170) / PRO 6000 (188) | gmean **2.04 / 2.15 / 2.21x** |

**On RTX PRO 6000** (the e2e part), H=64: `B=1..128` → 2.20 / 2.20 / 2.20 / 2.16 /
2.22 / 2.00 / 1.69 / 1.29x. **B=1..16 is 94% of the decode steps the e2e run
issued** (`#running-req`: 2:5683, 8:1387, 1:1223, 16:1176, 4:402; B=2 alone 54%).

## 2. Kernel — prefill · RTX 5090 · eager · speedup vs baseline

| tokens | H=8 | H=16 | H=32 | H=64 |
|---|---|---|---|---|
| 512 | 1.78x | 1.67x | 1.34x | 1.20x |
| 1024 | 1.74x | 1.59x | 1.30x | 1.37x |
| 2048 | 1.82x | 1.63x | 1.43x | 1.28x |
| 4096 | 2.05x | 1.77x | 1.36x | 1.20x |
| 8192 | 1.98x | 1.78x | 1.33x | 1.18x |

Flat in token count; narrows with head count. At DSv4's padded **h=64: 1.18–1.37x**.

## 3. End to end · 8× RTX PRO 6000 · SGLang `xutizhou/deepseek-base-optimization` @ `2538def09`

Config **verbatim, unchanged**:

```
--tp 4 --dp 4 --enable-dp-attention --ep 4 --pp-size 1 --moe-runner-backend deep_gemm
--chunked-prefill-size 4096 --mem-fraction-static 0.70 --disable-flashinfer-autotune
--disable-radix-cache --skip-server-warmup --cuda-graph-max-bs 16
--dsa-paged-mqa-logits-backend deepgemm --enable-deepseek-v4-fp4-indexer
```

Arm A = config as-is · Arm B = + the two switches · disjoint 4-GPU groups · 3 reps ·
**run twice with arms swapped between groups** (jobs 3661108 / 3661109).

### 3a. Serving — OSL 1024, 64 prompts, rate inf, seed 3 · B/A ratio

| ISL | CC | output tput ↑ | TTFT ↓ | TPOT R1 ↓ | TPOT R2 ↓ | **TPOT mean ↓** |
|---|---|---|---|---|---|---|
| 1024 | 8 | 1.031 | 1.002 | 0.966 | 0.975 | **0.970** |
| 1024 | 32 | 1.034 | 1.024 | 0.959 | 0.968 | **0.964** |
| 1024 | 64 | 1.039 | 1.015 | 0.953 | 0.962 | **0.958** |
| 8192 | 8 | 1.026 | 1.016 | 0.962 | 0.975 | **0.968** |
| 8192 | 32 | 1.013 | 1.019 | 0.964 | 0.982 | **0.973** |
| 8192 | 64 | 1.001 | 1.024 | 0.972 | 0.998 | **0.985** |

**TPOT −1.5% to −4.2%, all six points.** Resolution of this experiment: the
placement swap alone moves TPOT **±1.4 pp**, so read the mean column.

### 3b. Decode alone (OSL-heavy) · B/A

| CC | tput R1 ↑ | tput R2 ↑ | TPOT R1 ↓ | TPOT R2 ↓ |
|---|---|---|---|---|
| 8 | 1.034 | 1.047 | 0.966 | 0.949 |
| 64 | 1.034 | 1.049 | 0.966 | 0.950 |

### 3c. Prefill alone (OSL 1, CC 1 — no queueing) · TTFT B/A

| ISL | R1 | R2 |
|---|---|---|
| 1024 | 0.983 | 0.997 |
| 8192 | 0.998 | 1.010 |

**Neutral** — straddles 1.0 in both placements. In-situ CUDA-event timing: the
kernel is **1.565x** in the server (A 484.5 µs kernel + 105.1 µs page transcode =
589.6 µs; B 376.7 µs), but attention is only **2.1% of TTFT at ISL 1024, 5.5% at
8192**, so 1.57x on it is worth ≤1.4% of TTFT. Recommend leaving the prefill
switch **off**.

---

## 4. Accuracy

### 4a. Model accuracy — GSM8K, 400 questions, 5-shot, temperature 0

| arm | R1 rep1 | R1 rep2 | R2 rep1 | R2 rep2 | mean | invalid |
|---|---|---|---|---|---|---|
| **A** (FlashInfer) | 0.963 | 0.963 | 0.953 | 0.965 | 0.961 | 0.0 |
| **B** (this kernel) | 0.960 | 0.963 | 0.958 | 0.968 | 0.962 | 0.0 |

**Indistinguishable** — B's range (0.958–0.968) sits inside A's (0.953–0.965).
Same model, same weights, same sampling; only the attention kernel differs.

### 4b. Numerical accuracy — vs fp32 oracle over the **dequantised stored** KV

| metric | value |
|---|---|
| cosine similarity | **0.999599** |
| relative L2 | **2.83e-2** |
| change from head tiling | none to 6 significant figures |

Oracle is built from the values the cache actually holds, so quantisation the
kernel never chose is not charged to it.

### 4c. Invariance

| property | result |
|---|---|
| head tiling within a layout family (h=16/32/64, all (B, splits), ragged rows) | **bitwise identical** |
| merge `num_warps` 1 vs 2, every (B, splits), two devices | **bitwise identical** |
| across BLOCK_H=64 boundary (`warpsPerCTA` `[1,4]`→`[4,1]`) | cos 1.0000000, max\|d\| 0.0010 — **< 1 bf16 ULP** |
| `splits=1` vs unsplit kernel | **bit-for-bit** |
| split-K (`splits>1`) | algebraically exact (log-domain), not bitwise — softmax reassociated |

### 4d. Test suite

**36/36 pass** on RTX PRO 6000, RTX 5080 and RTX 5090 — under the default merge
policy *and* under merge `num_warps` forced to 1, 2, 4 (jobs 3660704 / 3660705 /
3659011). 1 skip: the FlashInfer cross-check, not importable in that container.

---

## 5. Hardware utilisation · RTX PRO 6000 · denominators measured on the part

Measured peaks: **fp8 749.9 TFLOP/s**, **bf16 392.5 TFLOP/s**, **1460 GB/s** copy.

| | achieved | % GEMM peak | % copy BW |
|---|---|---|---|
| decode B=1 | 6.8 TFLOP/s | 0.90% | 2.0% |
| decode B=16 | 57.2 TFLOP/s | 7.63% | 17.2% |
| decode B=128 | 80.7 TFLOP/s | 10.76% | 24.3% |
| prefill T=4096 | 109.1 TFLOP/s | 27.81% (bf16) | — |

Arithmetic intensity **227.6 FLOP/byte** vs ridge point **513.5** → the shape is
**memory-bound**; low FLOP fraction is expected, not a defect. Fixed by DSv4's
`d=512, topk=640`. At B=1–16 neither bound binds — the launch is 40 blocks
against 376 slots (**11% of the device**), i.e. a parallelism limit.

**Ceiling:** attention is **2.4–7.5%** of a decode step (derived from the A/B:
`share = (1 − TPOT_B/A) / (1 − 1/s)`). An infinitely fast attention kernel buys
−7.5% TPOT at best. Measured −1.5 to −4.2%.

---

*Full method, design, caveats and rejected experiments: `README.md` /
`docs/DSV4_FORK_INTEGRATION.md`.*
