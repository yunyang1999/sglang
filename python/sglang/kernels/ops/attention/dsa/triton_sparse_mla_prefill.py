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

End to end, served, 8x RTX PRO 6000 Blackwell (sm_120), TP8
------------------------------------------------------------
The real 150 GB DeepSeek-V4-Flash, both arms serving, `exit=0`. Arm A is the
stock SM120 route -- FlashInfer `_sparse_mla_sm120` with heads padded 8 -> 64.
Arm B routes prefill here with 8 real heads and no padding, and is charged for
the workspace build it needs.

    input_len   A TTFT ms   B TTFT ms   A/B      A tok/s   B tok/s
    2048         293.10      254.25     1.153x    6987      8055
    4096         574.74      491.64     1.169x    7127      8331
    8192        1130.81      961.73     1.176x    7244      8518

Run-to-run spread is <= 3 ms against gaps of 39-169 ms, and the advantage grows
with length. **Roughly a third to a half of the kernel-level win survives to end
to end** (1.15-1.18x against 1.5-2.3x); the rest is amortised across MoE, the
indexer, and arm B's workspace build.

Which kernel ran is measured, not inferred -- per-call counters inside the
dispatch, max across TP ranks:

    counter                          arm A            arm B
    sparse_mla_prefill (this file)       0            9050 @ heads=8
    sm120_attn_flashinfer            10696            1548  (decode)
    flash_mla_with_kvcache_sm120     10697 @ h=64     1548  @ h=64
    flash_mla_with_kvcache_cuda          0               0

Decode was a **harness control, not a result**: both arms ran identical decode
wiring, and TPOT matched to within 0.6% at every batch (11.85 / 15.0 / 28.8 /
45.2 ms at batch 1 / 8 / 32 / 64). That is what licenses trusting the prefill
delta -- the harness resolves no spurious difference.

**Model accuracy: cleared on the measurement that matters.** GSM8K, 400
questions, 5-shot, temperature 0, two repetitions per arm, same servers:

    arm A (FlashInfer baseline)   0.968  0.953   mean 0.9605   invalid 0.000
    arm B (this kernel)           0.960  0.970   mean 0.9650   invalid 0.000

The spread *within* each arm (1.5 and 1.0 points) is larger than the difference
*between* them (0.45 points), and this kernel is nominally the higher of the
two. vLLM publishes 95.0% for DeepSeek-V4 on this hardware class, so both arms
are healthy. Four runs, zero invalid outputs.

Two things that are true alongside that, and are not swept up in it:

* At temperature 0 the arms' greedy continuations separate after 4, 5 and 26
  tokens. That is expected of two different numerical paths and says nothing on
  its own -- token agreement with arm A would be the wrong test in any case,
  since against an fp32 oracle *this* kernel is the more accurate of the two
  (0.9999979 against FlashInfer's 0.9998678).
* A teacher-forced logit comparison over three fixed continuations gives top-1
  agreement of 16/16, 22/24 and 46/47, with mean KL(A||B) of 0.09, 0.19 and
  0.05. But the three disagreements sit at top-2 margins of ~0.11-0.12, not at
  near-ties. So the two distributions genuinely differ rather than merely
  flipping coin-tosses -- which is unsurprising when one arm pads 8 heads into
  64 and the other does not, and when both are approximations that differ from
  fp32 in different directions. It does not translate into a task-quality
  difference, and the oracle comparison says the baseline is the one further
  from truth, but it is recorded rather than explained away. The sample is
  small (16-47 positions on three low-constraint prompts); a wider logit study
  would be the way to close it properly.

Acceptance measurement, SM120, both precisions, both stages
------------------------------------------------------------
RTX 5080, DeepSeek-V4 shape (d=512, swa 128 + c4 512, h=8 after TP8, attn_sink),
against FlashInfer reached through SGLang's own ``_flash_mla_flashinfer`` with
the page split included. Our columns charge the per-layer workspace build
(``prep``) against us. Baseline is FlashInfer's narrowest legal instantiation
for the stage -- h=16 for prefill (its multi-group kernel refuses 8), h=8 for
decode (its decode dispatch takes 8), so neither side is padded at decode.

    prefill   fi h=64   fi h=16   ours bf16   ours fp8   prep    bf16    fp8
    T=512      0.931     0.313      0.179      0.147    0.026   1.52x  1.80x
    T=2048     3.480     1.216      0.619      0.506    0.026   1.89x  2.28x
    T=4096     6.928     2.390      1.235      1.016    0.027   1.90x  2.29x
    T=8192    13.651     4.746      2.454      2.017    0.027   1.91x  2.32x

The decode rows first published here were measured eagerly and were wrong; they
are replaced below. FlashInfer's wrapper carries ~106 us of host overhead --
0.124 ms eager against 0.0205 ms under CUDA-graph replay, which is what an
SGLang decode step actually pays. This kernel measures ~0.039 ms either way, so
eager timing flattered us by roughly 6x on that arm. Prefill is unaffected:
prefill is not graph-captured, so eager is the right methodology there, and at
T >= 2048 the 106 us is under 2% of the baseline (it does inflate the T=512 row).

Decode, re-measured under CUDA-graph replay, with a split-K path added
(`_nsa_decode_split_kernel` + `_nsa_decode_merge_kernel`, grid (T,S) then (T,H);
S=1 dispatches the existing kernel unchanged and is bitwise identical to it):

    B    fi h=8    ours unsplit   ours split-K   split vs fi   fp8 split vs fi
    1    0.0205      0.0390         0.0082          2.50x          1.43x
    2    0.0205      0.0391         0.0082          2.49x          1.33x
    4    0.0205      0.0389         0.0087          2.36x          1.25x
    8    0.0206      0.0391         0.0102          2.01x          0.77x
    16   0.0266      0.0409         0.0123          2.16x          0.57x
    32   0.0410      0.0410         0.0205          2.00x          0.45x
    64   0.0615      0.0418         0.0348          1.77x          0.40x

Three things this says that the eager numbers hid:

* **The unsplit path loses under graphs at B <= 32** (0.53x at B=1). One program
  per query token leaves 1 SM of 84 busy; nothing about that is fixed by the
  per-token maths being good. Split-K is what makes decode competitive, and its
  win is real GPU time rather than recovered launch overhead.
* Split-K does not fill the device either -- at B=1 it lights 10-20 SMs of 84,
  because 84 splits of topk 640 would be under 8 candidates each. The gain is a
  shorter serial chain, not occupancy.
* **The paged-fp8 arm goes 0.12x -> 1.43x at B=1 (11.8x its own unsplit form)
  and still loses above B=4.** Splitting hides the gather deficit at small
  batch; it does not fix it. That deficit is the remaining work.

Split-K over the *native* paged-fp8 gather closes the last gap. The table above
used the old converting gather; re-driven by the native one, in the same run:

    B    fi h=8   native unsplit   native split-K   S    fi/split   cos
    1    0.0205      0.0430           0.0103       10      2.00x   0.9995876
    8    0.0205      0.0430           0.0102       10      2.00x   0.9996073
    16   0.0250      0.0430           0.0123       10      2.03x   0.9995874
    32   0.0392      0.0431           0.0184        5      2.13x   0.9995892
    64   0.0611      0.0445           0.0348        2      1.75x   0.9996021

**There is no crossover.** The old gather's arm, measured beside it, falls under
1.0x between B=4 and B=8 (1.43x, 1.32x, 1.26x, 0.81x, 0.53x, 0.43x, 0.39x); the
native one holds 1.75-2.13x everywhere FlashInfer runs at all. fp8 decode is now
within 25% of bf16 at B <= 8 and identical at B=16 (0.0123 both), against 3-5x
behind before. The split kernel keeps the fp8 tensor core: 112x
``mma.sync.aligned.m16n8k32...e4m3.e4m3.f32`` plus 32 bf16 for the rope dots,
48384 B shared, against the old gather's 0 fp8 mma and 90368 B.

Two integration facts that will bite whoever ships this:

* **``_config``'s pinned ``h <= 8`` tile (32,4,3) is the bf16 kernel's tile and
  costs the native fp8 path 1.48x** (0.0635 against 0.0430 ms unsplit). The fp8
  arm needs its own (64,4,2) with BLOCK_H 8; it must not inherit ``_config``.
* The split heuristic's ``_MAX_WAVES`` does **not** carry over from the old
  gather: 4 -> 1. A gather ~6x cheaper makes a split's fixed cost relatively
  larger, so programs stop paying a wave earlier. ``_MIN_CHUNKS_FP8`` does carry
  over at 1, steeply (2 costs 1.14x, 4 costs 1.80x).

Split-path accuracy: cos >= 0.9999956 at every (B, S) measured, max_abs 0.00195
= exactly one bf16 ULP for outputs in [0.5, 1), i.e. it differs from the unsplit
reference by at most the output format's own quantum. fp32 partials move cos to
0.9999974 and do not move max_abs, so bf16 partials are enough.

Accuracy, against an fp32 oracle over the values the KV cache actually holds --
production stores fp8 and both kernels read it, so that is the only reference
under which "no accuracy impact" means anything:

                    prefill      decode
    ours bf16      0.9999980   0.9999979
    FlashInfer     0.9998678   0.9995651
    ours fp8       0.9994601   0.9994475

The bf16 path is roughly 7x *more* accurate than FlashInfer (1.3e-6 against
1.3e-4). The fp8 path is somewhat less accurate than FlashInfer's (5.4e-4), and
the reason is structural rather than inherent: it re-quantises with a global
amax a tensor that was already fp8-derived, i.e. it quantises twice, where
FlashInfer consumes the stored fp8 and its ue8m0 scales directly. Reading the
paged cache natively removes that second quantisation entirely.

Two limits on these numbers, stated rather than buried:

* The ``ours`` arms read a flat workspace built from the paged cache. Prefill
  already has one (``_forward_prefill_sparse`` builds it); **decode does not**,
  so the decode rows describe a layout the decode path would have to be given.
  The ``prep`` charge is a whole-pool dequantise, which at decode is far more
  than the gathered rows would cost -- conservative, not optimistic.
* Scale granularity is not what limits fp8 here. Measured at this shape, global
  amax, per-row and per-64 ue8m0 land within 2e-5 of each other against fp32
  (0.9991758 / 0.9991775 / 0.9991563). The double quantisation is the cost, not
  the granularity.

Attention subsystem per forward, 43 layers -- a proxy for end to end, because
DeepSeek-V4 is 150 GB across 8 GPUs and does not fit on any single SM120 part:

    T=4096:  SGLang today 297.9 ms | FI h=16 102.8 | ours bf16 54.2 | fp8 44.8
    T=8192:  SGLang today 587.0 ms | FI h=16 204.1 | ours bf16 106.7 | fp8 87.9

Reading the paged fp8 cache natively: it works, and it still loses to bf16
-----------------------------------------------------------------------------
``sparse_mla_prefill_paged_fp8`` in this file converts the KV tile in registers
and therefore never reaches the fp8 tensor core (its compiled kernel emits 128
bf16 mma and 90368 B of shared memory, i.e. one block per SM). A rewritten
version that never constructs the tile -- 448-wide nope loaded as 7 chunks of
64, one ue8m0 scale each, handed to ``tl.dot`` unconverted; the 64-wide bf16
rope tail given its own dot; the KV scale folded into P before P is quantised --
does reach it: 56x ``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`` and
25344 B shared, **6.21x faster than the version here** (8.19 -> 1.32 ms at
T=4096).

    T       FI h=64   FI h=16   native fp8   ours bf16   vs FI16   vs bf16
    512      0.918     0.309      0.213      0.179+.019   1.45x     0.84x
    2048     3.439     1.195      0.686      0.606+.019   1.74x     0.84x
    4096     6.867     2.353      1.364      1.239+.019   1.73x     0.83x
    8192    13.637     4.654      2.705      2.465+.019   1.72x     0.84x

So it clears the bar against FlashInfer and still loses to this file's own bf16
path. The reason is structural rather than tuning, and it is worth stating
because it bounds what fp8 can do here in Triton at all: DeepSeek-V4's ue8m0
scale varies **along** QK's reduction axis (one per 64 of the 448), so it cannot
be hoisted out of a single dot, and Triton cannot slice a tensor -- all seven
chunk tiles must stay live from the gather through both dots. Ablated at
T=4096: 12 extra dots cost 0.133 ms, **6 extra gathers cost 0.853 ms**. The
gather is ~65% of runtime and the fp8 tensor core accelerates the ~10% that is
dots. FlashInfer pays neither, with hand-written ``ldmatrix`` against an
explicit smem layout.

Accuracy: cos 0.9996209 against an fp32 oracle over the dequantised stored KV,
against 0.9999979 for bf16. **The KV side contributes exactly zero error** -- the
stored bytes go to the MMA unconverted. 72% of the loss is Q quantisation, which
is unavoidable for this path: Triton rejects a mixed bf16 x fp8 dot outright
(``Unsupported rhs dtype fp8e4nv``), so using the fp8 tensor core means
quantising Q. "No additional accuracy loss" therefore holds on the KV side only.

One frontend note worth recording: ``tl.dot_scaled`` does lower natively on
sm_120 to the hardware block-scaled instruction FlashInfer hand-writes, but
**only when both scale operands are real tensors**. With ``lhs_scale=None`` it
silently takes the documented bf16-upcast emulation (96 bf16 mma, 0.28x speed).
Pass an all-127 (2^0) lhs tile if only the B side needs scaling.

What it is actually faster than, on DeepSeek-V4 / SM120
--------------------------------------------------------
An earlier revision of this note quoted a very large speedup against
``flash_mla_sparse_decode_triton``. That number was against the wrong kernel and
is withdrawn. ``environ.py`` sets ``SGLANG_SM120_FLASHMLA_BACKEND`` to
``"flashinfer"``, and ``flash_mla_sm120.py`` describes the other values as
forcing a fallback, so a stock SM120 deployment runs **FlashInfer's**
``_sparse_mla_sm120`` (available from flashinfer-python 0.6.14; SGLang imports it
without a try/except, so an older FlashInfer raises rather than falling back).

The production call also does not have the shape this kernel takes. SGLang
passes the sliding window (topk 128) as ``indices`` and the indexer selection
(topk 512) as ``extra_indices`` -- two pools, never concatenated -- while this
kernel takes the single combined 640 over the one flat workspace
``_forward_prefill_sparse`` builds.

And FlashInfer's SM120 prefill dispatches from a fixed table. Probing 70
(heads, topk) pairs through SGLang's own ``_flash_mla_flashinfer``, exactly 16
are accepted:

    heads in {16, 32, 64, 128}  x  topk in {128, 512, 1024, 2048}

DeepSeek-V4 after TP8 has **8** heads and a combined topk of **640**, so its real
shape misses that set on both axes. That is what ``models/deepseek_v4.py``'s
``padded_num_heads = 64 if n_local_heads <= 64 else n_heads`` is for: the padding
is not a tuning choice, it is the only way to make the call legal.

Where the gap actually is, and what it is worth -- this corrects an earlier
version of this note, which claimed an upstream heads=8 instantiation would be
worth more than anything in this file. It would not.

The refusal is real and was re-checked on the dual-cache DSv4 path with the
extra pool at the required 584 B/token. Its cause is specific: DSv4 shapes go to
the *multi-group* kernel, and ``prefill_kernel.cuh`` hard-asserts
``NUM_HEADS % (MG_N_HG_T * HPB) == 0`` with no ``VALID_HPB`` gating. The
single-group kernel does implement the documented "NUM_HEADS < HPB=16 zero-pads
+ gates" and does accept 8, but only DSV3_2/GLM_NSA reach it. Decode is
different again: its dispatch holds (8,128), (8,512), (8,1024) outright.

But an h=8 instantiation would buy ~nothing over h=16, because 8 heads still
fill a 16-row mma tile. Taking FlashInfer's own two measured points on this
device -- 4.66 ms at h=16, 13.68 ms at h=64 for the same work -- and solving
``T16 = G + M``, ``T64 = 2(G + 2M)`` gives a per-CTA gather term G ~ 2.48 and a
maths term M ~ 2.18, i.e. **h=8 would cost what h=16 costs**.

The lever is one step earlier, in SGLang: ``models/deepseek_v4.py`` pads to 64
when 16 would be legal and sufficient. Padding 8 -> 16 instead of 8 -> 64 on the
SM120/FlashInfer route is three lines and worth **2.94x on the baseline**
(13.68 -> 4.66 ms, both measured here). Stating the consequence plainly: it also
cuts this kernel's prefill advantage from 5.6x to 1.9x. That is the honest
number, and it is better found here than by a reviewer.

Measured that way -- FlashInfer in production form (8 real heads padded to 64,
swa 128 + extra 512, page-split included) against this kernel with 8 real heads,
RTX 5080:

    B      A: flashinfer   C: ours paged fp8   D: ours bf16 + dequant   A/C    A/D
    128       0.248 ms          0.323 ms            0.051 + 0.023        0.77x  3.38x
    1024      1.707 ms          2.083 ms            0.308 + 0.022        0.82x  5.18x
    8192     13.323 ms         15.792 ms            2.371 + 0.022        0.84x  5.57x

Two honest readings, and they point opposite ways:

* **On the same paged fp8 pool this kernel loses**, 0.77-0.84x. FlashInfer's
  hand-written CUDA reads that layout better than the Triton gather does, and it
  wins even while doing 8x the head work. The paged-fp8 entry here is not
  competitive and should not be presented as if it were.
* **On the flat bf16 workspace it wins 3.4-5.6x**, dequant charged in. But that
  is a claim about the *route*, not the kernel: it says the sparse-prefill path
  (dequantize once, gather bf16, no page transcode, no head padding) beats the
  decode-entry path on SM120 -- which is what enabling
  ``SGLANG_DSV4_TRITON_SPARSE_PREFILL`` does.

For scale, FlashInfer's own head-padding tax is visible in the same run: 8 real
heads padded to 64 costs **2.5-3.1x** the same work at its narrowest
instantiation (16). Removing that tax is this kernel's whole structural
advantage, and it is worth roughly what the padding costs -- not more.

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

Why head-tiling actually works, and the 128 bytes that would pay next
-----------------------------------------------------------------------
The account given below for head-tiling -- that the `[BLOCK_H, d_v]`
accumulator pins occupancy to 1 block/SM -- is **wrong about the mechanism**,
though right about the root cause. Nsight Compute on both arms at h=16 and h=64
says so directly: the *winning* configuration has **half** the occupancy of the
losing one (1.001 warps/scheduler at h=16 against 2.001 at h=64), so occupancy
cannot be what separates them.

What separates them is that **the monolithic h=64 kernel issues exactly 2.5x
the tensor ops the algorithm needs** -- 214,748,364,800 against a requirement of
85,899,345,920, while the h=16 kernel issues exactly 1.000x. The cause is a
Triton mma-layout artefact: `_PINNED_WIDE_H` gives BLOCK_N=16 with 8 warps, the
QK dot's N=16 is only two mma n-tiles, and eight warps at `instrShape [16, 8]`
span 64 -- so eight warps compute two n-tiles **four times over**. Confirmed
three independent ways: 160 `mma.sync` in the PTX where 128 suffice, 128 static
HMMA PCs at the QK-dot line against 32 at the PV-dot line, and the counter
matching 1280 x 40 x 1024 x 4096 to the last digit. The PC histogram shows it
costing real time -- the two dots carry identical FLOPs, and at h=16 they cost
21.2% and 16.1% of issue-stall samples, but at h=64 the QK dot costs 4.3x the PV
dot (59.3% against 10.7%), with `math_pipe_throttle` at 57.9%.

The accumulator is still the root cause, one step removed. At BLOCK_H=64 it is
64*512*4 B, which over 128 threads is 256 registers per thread against a 255 cap
-- the 4-warp cubins duly carry a 432-480 B stack frame. So 8 warps are forced;
and with 8 warps, avoiding the replication needs the QK output N >= 64, i.e.
BLOCK_N >= 64, which needs 139,520 B of shared memory against a 101,376 B
budget. **On SM120 there is no BLOCK_H=64 tile that is simultaneously
non-spilling and non-replicating.** That is why the 96-config sweep found
nothing and why the grid was the only axis left. Removing 60% of the tensor ops
in a kernel 57.9% math-throttled predicts 1.6-1.8x; the delivered 0.67 -> 1.20
is 1.79x.

**The "next lever is 128 bytes" claim was wrong, and is withdrawn.** It read the
byte budget off the TTGIR `local_alloc` ops -- Q 16,384 + KV 32,768 + P 1,024 =
50,176, plus a 128 B index tile -- and concluded the BLOCK_H=16 / BLOCK_N=32 /
4-warp cubin missed 2 blocks/SM by exactly that index buffer. Compiling the
kernel for sm_120 ahead of time and reading `metadata.shared` says the cubin
actually carries **57,344 B** at num_stages=1 and 57,472 at 2. The three named
allocations do sum to 50,176; the other **7,168 B** is compiler scratch that no
`local_alloc` names. So the gap to the real 2-block threshold (101,376 / 2 =
50,688 B) is 6,656 B, not 128, and no amount of index-buffer surgery closes it.
The rest of that profile still holds -- shared memory *is* the sole occupancy
limiter, and the mma count *is* already the algorithmic minimum -- which is why
the bf16 arm has no cheap occupancy win left. Every mma-minimal bf16 tile on
sm_120 (H16/N32, H16/N64, H32/N32, H32/N64, H64/N64) is above the threshold; the
two tiles that fit (H16/N16 at 36,928 B, H8/N32 at 49,280 B) buy their second
block with 1.50x and 2.00x the tensor work respectively.

**Where the occupancy win actually was: the native fp8 arm, which never got head
tiling.** `_nsa_prefill_paged_fp8_native_kernel` indexed heads as
`tl.arange(0, BLOCK_H)` on grid `(T,)`, and `block_h = max(_NATIVE_BLOCK_H,
next_pow2(h))` pinned that to 64 at DeepSeek-V4's padded h=64 -- the same
monolithic shape this file had just finished removing from the bf16 arm. Its
sm_120 cubin at the pinned BLOCK_N=64:

    BLOCK_H  shared     blk/SM  warps/SM  spill   tensor work
    64       82,176 B   1       4         9,104 B  1.00x   <- was shipping
    32       63,744 B   1       4         2,912 B  1.00x
    16       54,528 B   1       4         1,320 B  1.00x   <- now pinned
    8        49,920 B   2       8           612 B  2.00x

The measured win is **1.75-2.88x on decode, gmean 2.12x over batch** (CUDA-graph
replay, RTX 5080, job 3624701) -- and note what it is *not*. At the inherited
BLOCK_N=64 the tile that ships now has the same 1 block/SM and the same 4
warps/SM as the one it replaces, and issues exactly the same number of mma. The
only thing that moved is the spill: **9,104 B down to 1,320 B**, because the
eight 64-wide fp32 accumulators are what overran the 255-register cap, and they
shrink with BLOCK_H. Occupancy is not the mechanism here; register pressure is.

That distinction was worth 17%. The first version of this change also narrowed
BLOCK_N to 32, which the shared-memory sweep said would buy 3 blocks/SM -- and
which measured *slower at every batch above 1*, taking gmean 2.12x down to 1.82x.
See `_PINNED_NATIVE_TILED_BN`, kept empty with the numbers.

BLOCK_H=8 is excluded deliberately: mma's M granularity is 16, so an 8-row tile
issues the same instructions for half the useful rows -- the one row in the table
that does buy occupancy, and pays double the tensor work for it. It survives only
where H itself is 8 and no wider tile is legal.

This is a pure indexing change, and it is bitwise **within a layout family**.
Triton gives a `[BLOCK_H, BLOCK_N]` tile a `warpsPerCTA` of `[1, 4]` up to
BLOCK_H=32 and `[4, 1]` at 64, which turns the softmax row reduction
`tl.sum(p, axis=1)` from a cross-warp tree into an intra-warp one -- same
arithmetic, different summation order. So 8, 16 and 32 agree bit for bit with
each other (verified at h=16/32/64, through the split-K decode kernel at every
(B, splits), and on ragged rows), while 64 sits on the far side of that boundary
at cos=1.0000000, max|d|=0.0010 -- under one bf16 ULP. What holds across the
boundary too, and is the property that matters, is that **tiling does not move
the error**: every tile lands on the same cos and max|d| against the bf16 gather,
to every digit printed, at every h.

An earlier version of this note blamed wgmma, whose M is fixed at 64 and which
does flip at exactly that BLOCK_H on sm_90 (168 wgmma, zero mma.sync, against
mma.sync-only below it). That is a second effect in the same place, not the
cause: sm_120 has no wgmma at all -- BLOCK_H 64/32/16 emit 576/288/144 mma.sync
-- and the boundary is still there. It was read off the TTGIR after the sm_120
run contradicted the guess, which is the only reason it is right now.

Narrowing BLOCK_N alongside the tile was tried and lost -- see
`_PINNED_NATIVE_TILED_BN`, kept empty with the numbers. Two things it left
behind. BLOCK_N is the softmax tile width, so changing it is not bitwise
(cos=0.999976, max|d|=0.0391), which means any tile-vs-tile comparison has to
pin `config` or it measures the BLOCK_N change and blames tiling -- exactly what
the first sm_120 run did, and why the tests now pass an explicit config. And the
shared-memory sweep that motivated it was right about the bytes and wrong about
the speed: 3 blocks/SM measured *slower* than 1 at every batch above 1. On this
kernel warps/SM is not the binding constraint an occupancy table makes it look
like -- the same lesson the bf16 arm got from NCU, arrived at twice.

Three levers the same profile **closes**, so they need not be retried: fp8 for
the QK dot (tensor subpipe only 25.5% active and math throttle 25.7% of samples,
so halving QK tensor cycles saves at most ~11% -- and the gather side of that
trade was already measured going the wrong way); the fp32 accumulator rescale
(`sm__pipe_fma_cycles_active` is **2.23%** of peak, NCU estimates 1.1%, so the
37% attributed to "PV accumulate" at h=8 does not survive at BLOCK_H=16); and
register pressure or `maxnreg` (204 registers, register limit already 2 blocks,
zero spills anywhere).

And two things about FlashInfer worth recording, because they invert the story
told below. **We amortise the gather better than it does, and still lost at
h=64**: from h=16 to h=64 its L2 read sectors grow 2.08x (it splits each token
across two blocks, gathering 1280 rows per token against our 640) while ours
grow 1.07x. The entire loss was the tensor pipe -- 3.20x its tensor cycles. What
it does have is **43.75% of its tensor ops on the fp8 pipe**, reading the stored
fp8 directly, where we issue zero fp8 tensor ops off a dequantised workspace. It
is otherwise the more wasteful kernel by a wide margin: 81.3M shared-memory bank
conflicts against our 676, 38-41% excessive global sectors against our 0, and it
runs 5.6x above its own tensor floor to our 4.2x.

On vLLM, and what it says about where the advantage comes from
----------------------------------------------------------------
Measured inside vLLM 0.27.1's own environment (torch 2.13.0+cu130, flashinfer
0.6.16.post3, its own pin) on one RTX PRO 6000, against the kernel vLLM
dispatches on SM120 -- FlashInfer's `_sparse_mla_sm120`, the same one SGLang
uses. Prefill, eager (vLLM does not graph-capture prefill), T=2048/4096/8192:

    heads (TP)   ours paged fp8   ours flat bf16   vs vLLM's own Triton
      8 (TP8)    FI refuses       FI refuses       2.87 - 10.1x
     16 (TP4)    1.29 - 1.36x     1.19 - 1.25x     2.18 - 8.6x
     32 (TP2)    0.71 - 0.74x     1.36 - 1.43x     2.13 - 11.4x
     64 (TP1)    0.44 - 0.46x     0.91 - 0.97x     1.22 - 6.6x

Decode, under graph replay: 1.29-1.60x at h=8, 1.34-1.56x at h=16, 0.80-1.08x
at h=32. Decode is the one place FlashInfer accepts 8 heads, so TP8 decode is a
real head-to-head rather than an enablement.

**vLLM does not pad the head count.** Its SM120 route pads to the smallest
member of `(8, 16, 32, 64, 128)` at or above the real count, and DeepSeek-V4's
64 heads divide evenly at every TP, so the factor is 1.00x throughout. The
largest single source of the SGLang win is therefore absent by construction
here, and what is measured above is the rest of the design.

**And at TP8 vLLM cannot prefill at all.** FlashInfer's prefill orchestrator has
no 8-head instantiation (`Unsupported sparse-MLA prefill configuration:
num_heads=8 ...`, reproduced at page_block_size 64 and 256 and on 0.6.17), and
vLLM 0.27.1 ships no Triton attention fallback -- `sparse_mla_kernels` is absent
from the wheel and the DSv4 selector returns the FlashInfer class unconditionally
on major==12. So at the TP the vendor recipe specifies for 8x RTX PRO 6000, this
kernel is an enablement rather than a speedup. If vLLM instead padded 8 -> 16
there (three lines, and legal), FlashInfer would cost 1.738 ms at T=8192 against
our unpadded 0.965 ms: 1.80x.

Accuracy, cosine against an fp32 oracle over the dequantised stored KV at h=16,
T=8192: **ours flat bf16 0.9999973** against FlashInfer's 0.9997166 -- about
100x closer to fp32 -- with our paged-fp8 arm at 0.9996866, consistent with the
double-quantisation account above. vLLM's own Triton kernel over the same
workspace also reaches 0.9999973, which is independent evidence the workspace
and index construction are right.

Two corrections to assumptions that were made here earlier. vLLM's SM120 route
does **not** hand us a flat `kv` plus combined indices -- that is its SM90/SM100
path; SM120 passes two paged pools and two index arrays, and building the flat
workspace costs 0.034-0.035 ms per layer, small but not free. And both DSv4
pools are **64 tokens per page**, not 256; FlashInfer's decode dispatch requires
64 and silently falls through to the prefill orchestrator at 256.

The advantage is a function of head count, and it reverses above ~32
--------------------------------------------------------------------------
Everything else in this file was measured at h=8, DeepSeek-V4's post-TP8 count.
At a wider head count the design inverts, and the crossover is not a tuning
artefact. Measured on an RTX PRO 6000 against FlashInfer's `_sparse_mla_sm120`
in production form (page transcode charged to it, our workspace dequantise
charged to us):

    heads   T=512   T=1024   T=2048   T=4096      <- ours / theirs
      16    1.21x    1.32x    1.35x    1.51x
      64    0.87x    0.84x    0.92x    0.96x

NCU at T=1024 names the mechanism: the `[BLOCK_H, d_v]` fp32 accumulator. At
h=64, `BLOCK_H=64` makes it 64x512x4 B over 256 threads = **128 registers per
thread of 246**, against a 255 cap, pinning the kernel to 1 block/SM. Our cost
per head *rises* 1.60x from h=16 to h=64; FlashInfer's *falls* 1.25x, because it
amortises one gather over more heads and splits each token across two blocks
(grid 2048 against our 1024). A full `BLOCK_N x warps x stages` sweep at h=64
found the pinned tile already optimal at every T, so this is structural.

**Tiling the head dimension across the grid fixes it, and is the one structural
change nothing here had tried.** Every experiment in this file holds
`BLOCK_H = H` and varies the tile; none varies the grid, which stays `(T,)`.
Running grid `(T, H/16)` -- four programs per token, each with a `[16, 512]`
accumulator -- trades a 4x re-gather (L2-resident, 77.6% hit) for a 4x smaller
accumulator. Emulated as four 16-head calls over the same KV and indices:

    T        theirs   ours monolithic   ours head-tiled(16)   vs theirs
    512      0.4073       0.4483              0.3926            0.99x
    1024     0.8188       0.9685              0.7117            1.12x
    2048     1.6285       1.7421              1.3260            1.21x
    4096     3.2176       3.4600              2.7768            1.15x

A 0.84x loss becomes a 1.12x win at h=64. Tile width 8 is worse (0.95x of
monolithic -- an 8x re-gather is too much); width 32 was not cleanly measured.
The 1.12x is emulated rather than implemented, so treat it as a strong signal
rather than a delivered number.

Two things this reframes. First, the head-padding advantage that dominates the
h=8 story is worth nothing here: at h=16 the 16-row mma tile is exactly full,
and at h=64 there is no padding to avoid. What remains is the page transcode we
skip, measured at 1.501 ms against FlashInfer's 2.627 ms kernel, i.e. ~36% of
their prefill attention time -- though that probe used uniformly random indices
so every page was transcoded, which flatters us. Second, a deployment using
SGLang's DP attention resolves to a *wider* head count, not narrower:
`--tp 4 --dp 4 --enable-dp-attention` gives `attn_tp_size = 1` and therefore
`n_local_heads = 64`, with the padding block gated off entirely.

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
* **swap-AB was built in CUDA and measured. The premise holds exactly; the
  payoff does not.** `mma.sync.aligned.m16n8k16` has N granularity 8, so
  transposing both dots (`S^T = K.Q^T`, `O^T = V^T.P^T`) puts h=8 on N with no
  padding. A hand-written sm_120a microbenchmark confirms it in SASS: **4.00
  HMMA per KV row against the standard orientation's 8.00**, independent of
  BLOCK_N and of the warp partition -- 42.95 GFLOP issued against 85.90, i.e.
  exactly the required figure. Core-loop speedup 1.46x at BLOCK_N=32 and
  **1.92x at 64**; registers halve (240 -> 120 with Q resident); cosine
  0.9999988 against an fp64 oracle.

  **The relayout everyone assumes is the blocker is free.** Transposing the
  `S^T` accumulator into a B operand costs 2 `movmatrix` per warp per iteration
  against 32 HMMA saved -- and a variant that stores P as `[kv][head]` and reads
  it back with `ldmatrix.x2.trans` needs **zero** `movmatrix` and measures
  identically (15.761 vs 15.794 ns/kv-row). Both ride free on a shared-memory
  round trip of P that *both* orientations already need, because the warps that
  produce S are not the warps that consume it. FlashInfer's own SM120 kernel
  does the same round trip. So the reason FA, FlashMLA and FlashInfer all stay
  in the standard orientation is not relayout cost.

  **But in a complete kernel it is worth 1.09x, not the 1.35-1.5x derived here.**
  Built both orientations over one skeleton (identical `LDGSTS` counts in SASS,
  differing only in HMMA 64 vs 32): swap-AB beats its matched control by
  **1.092x at T=4096 and 1.089x at T=8192**, reproduced on a second SM120 part
  at 1.046x/1.106x. Solving the two arms gives an mma-structured core of 0.378
  ms against 1.058 ms of everything else -- **the dots are only ~26% of the
  kernel**, so halving them caps at 1.09x.

  The derivation was wrong for an instructive reason: it assumed the 2.00x
  tensor-FLOP overhead translated into runtime at the ~44% the profile
  attributes to the dots. It does not, and this file already said why -- "the PV
  side is not paying for arithmetic, it is paying for the wide `[BLOCK_H, d_v]`
  fp32 accumulator". A hand-written kernel fixes that accumulator *regardless of
  orientation*; once fixed, the mma instructions are a quarter of the kernel.

  The hand-written kernel reaches 0.895x of this Triton one at T=4096
  (1.3146 ms against 1.1761), losing entirely on its gather (1.058 ms against
  Triton's ~0.66). Precomputing row pointers at index-staging time moved it
  0.830x -> 0.895x, so that axis is live, but the swap's own contribution stays
  ~1.09x either way. Sources at `probe/../swapab/{phase1,phase2}.cu`.

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

Nsight Compute adds two facts the sweep could not see, and closes two more
candidates that follow from them:

* **``num_stages`` was inert.** The compiled TTGIR allocates
  ``memdesc<1x32x512xbf16>`` for the KV tile -- stage depth 1 -- because the
  gather address depends on the index load, so Triton's pipeliner cannot
  prefetch across it. The whole 96-config sweep therefore explored a ``stages``
  axis that did nothing. The cost is visible: the hottest single instruction in
  the kernel is ``BAR.SYNC.DEFER_BLOCKING``, 98% of its samples in
  ``long_scoreboard``, 5.15% of all samples.
  Breaking the dependency by hand (load iteration k+1's indices before consuming
  k's) does make Triton double-buffer -- and that is exactly why it loses:
  shared memory goes 41728 -> 74240 B, which is 1 block per SM instead of 2.
  Measured 0.74-0.81x at T=2048/4096/8192, bitwise-identical output. The second
  block is worth more than the pipelining.
* **The redundant half of the KV load mask is not free to remove either.**
  ``valid[:, None]`` is provably redundant (``row`` is clamped, and ``qk`` is
  forced to -inf on those columns so ``p`` is exactly zero), and line 237 spends
  ~48% of its samples on integer/predicate work. Dropping it measured 0.95x --
  slower, bitwise-identical. Recorded so it is not retried.
* Register pressure is 183/thread, of which 64 are the ``acc`` fragment and 32
  of those 64 are pure M-padding (16 HMMA accumulator destinations, 64 matching
  ``FMUL``s for ``acc * alpha``). Recovering *all* of them would still buy zero
  blocks: ``launch__occupancy_limit_registers`` and
  ``launch__occupancy_limit_shared_mem`` are both 2, and 32768 of the 41728
  shared bytes are one BLOCK_N=32 KV tile. ``maxnreg`` is not a lever here.
* The tensor pipe issues **85.90 GFLOP against 42.95 GFLOP required -- exactly
  2.00x** -- at 70.4% utilisation, with L2 behind it at 52.3%. Both numbers come
  from the same fact: each query token amortises its own 640-row gather over
  only 8 head rows. The profile's floor for this algorithm is ~0.59 ms, 1.9x
  today, reachable only by sharing the gather.

``union``: the retention question, answered from the real model
----------------------------------------------------------------
``union``'s payoff was gated on one number nobody had measured -- how much of
its selection a query token inherits from its predecessor. Captured from
DeepSeek-V4-Flash itself (8xH200, TP8, two real 16384-token prompts, all 43
layers, 176 index tensors), rather than modelled:

    positions      selection density   retention   union ratio at G=4
    0 - 2k         >= 1.0 (no choice)     1.000          1.01
    3k - 4k            0.50               0.85           1.18-1.23
    7k - 8k            0.25               0.81-0.85      1.27-1.39
    11k - 12k          0.17               0.68-0.76      1.47-1.59
    15k - 16k          0.12               0.57-0.73      1.42-1.80

Against the payoff curve measured here (0.50 -> 0.84x, 0.75 -> 1.06x,
0.90 -> 1.25x), that lands as:

* **positions 2k-8k: union G=4 wins, ~1.16-1.20x.**
* **positions 8k-16k: break-even.**
* **mid-network layers (L16-L34) past 15k: retention 0.46-0.57, union loses.**

And the direction is the problem. Retention tracks selection density, which
falls as context grows; DeepSeek-V4 advertises 1M context, so a fixed ``G=4``
gets *worse* with longer sequences, not better. It must not be enabled globally.
``G=2`` stays inside 1.12-1.28 union ratio everywhere measured -- but this
kernel's own G=2 timings only reach break-even at retention 0.90, so it is safe
without being profitable. The honest position: union is a shallow-context,
early/late-layer optimisation on this model, and there is no fixed G that is
right for a whole forward.

Two structural facts from the same capture, worth having:

* **There is no single [T, 640] index tensor.** The kernel is handed
  ``indices`` [T, 128] into the sliding-window pool and
  ``extra_indices_in_kvcache`` [T, 512] into the 4x-compressed C4 pool -- two
  address spaces. 640 is the total gather width, not one set. Only the 21 C4
  layers carry a selection at all; 20 C128 layers get a deterministic dense
  enumeration (retention exactly 1.000) and 2 layers are sliding-window only.
* The sliding window is exactly a shift-by-one: ``swa[t+1, j+1] == swa[t, j]``
  held on 1,032,256 of 1,032,256 valid pairs. Unioning that half is nearly free
  (1.023 at G=4) and dilutes the C4 cost by about 20%. No row names a KV
  position twice, in either set, anywhere in the capture.

An operational limit found the same way, unrelated to this kernel but worth
recording: DSv4 attention allocates ``sizeof(int) * (batch*5+1)`` of dynamic
shared memory for its scheduling metadata, and prefill puts one row per query
token, so it hard-caps at about **11,622 query tokens per forward** against
Hopper's 227 KB. Any ``chunked_prefill_size`` above ~11.6k fails with an
``invalid argument`` from ``cudaFuncSetAttribute``.

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
path, no cross-token sharing, so a decode step is simply ``T = batch``.

Against FlashInfer at the same head count — its *decode* dispatch does have
heads=8 (``_DECODE_DSV4_DISPATCH`` contains (8,128), (8,512), (8,1024)), unlike
its prefill — swa 128 + c4 512, RTX 5080:

    B    FlashInfer h=64   FlashInfer h=8   ours paged fp8   ours bf16
    1        0.1234 ms        0.1239 ms       0.1683 ms      0.0386 ms
    16       0.1246           0.1212          0.1687         0.0389
    32       0.1224           0.1226          0.1688         0.0390
    64       0.3525           0.1240          0.1695         0.0405

Three things follow, and the third is the one that matters:

* **SGLang's head padding is free below batch 64 and expensive at it.**
  ``models/deepseek_v4.py`` pads 8 -> 64 under ``if attn_tp_size > 1:``, with no
  architecture and no prefill/decode condition. At B <= 32 that costs nothing
  (1.00-1.03x) because the regime is latency-bound; at B = 64 it costs **2.84x**.
* **On bf16 this kernel is 3.06-3.28x faster than FlashInfer at heads=8**, flat
  across the whole decode range.
* **On paged fp8 it is 0.72-0.77x — it loses.** And decode has no flat bf16
  workspace: ``_forward_prefill_sparse`` builds one, the decode path does not.
  So the honest decode number today is the losing one, and the 3.2x is on a
  layout decode would first have to be given.

That makes the paged-fp8 gather — not the tile, not ``union`` — the thing worth
fixing. It is the same deficit on both stages (0.83x at prefill B=8192, 0.72x at
decode) and it is what stands between the bf16 results and the layout the KV
cache actually stores.

Separately, the design is still the wrong shape for small-batch decode and the
numbers show it: one program per query token means batch 1 lights up 1 of 84
SMs, and wall time is flat at ~0.039 ms from batch 1 to 64. Filling the device
there needs splitting one token's top-k across CTAs and merging — exactly the
partials-and-merge structure this kernel exists to avoid. It wins at these sizes
anyway because everything in the regime is latency-bound, not because the shape
suits it.

Known gap: the two paged-fp8 device helpers (``_paged_fp8_row_tile`` and
``_native_tile``) still test only ``idx >= 0``. Their bound is a token count
rather than a row count and neither their caller kernels nor their launches
carry it, so closing it is a wider change than the flat-KV path was. Both are
reached only through the opt-in ``sparse_mla_prefill_paged_fp8*`` entries, not
the default path. Until they are fixed, a caller that can produce an index at
or past its pool must clamp before calling them.

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
    n_rows,  # rows in the KV pool; an index at or past it is invalid
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
    # Head tile. grid dim 1 is ceil(H / BLOCK_H): one program per (token, head
    # tile) rather than per token. Every use of `h` below is an ABSOLUTE head
    # index -- into q, into the sink, into the output -- so offsetting it here
    # is the whole change. With BLOCK_H >= H the grid dim is 1, `hb` is 0, and
    # `h` is exactly `tl.arange(0, BLOCK_H)` as before, which is what makes the
    # monolithic form bitwise identical to the untiled kernel.
    #
    # The trade: the `[BLOCK_H, D_V]` fp32 accumulator shrinks with BLOCK_H (at
    # H=64 it is 128 of the kernel's 246 registers/thread and pins occupancy to
    # 1 block/SM), at the cost of re-gathering each token's KV rows once per
    # head tile. The gather is L2-resident, so on DSv4's shape that trade pays.
    hb = tl.program_id(1)
    D_TAIL: tl.constexpr = D_QK - D_V

    if FP8:
        # inputs pre-scaled by 448/amax in the wrapper; undo inside the math:
        # qk_real = qk_fp8 * qs*ks/448^2 ; pv_real = pv_fp8 * ks/448^2 (P carries x448)
        qk_scale = sm_scale * tl.load(scale_ptr) / (448.0 * 448.0)
        out_scale = tl.load(scale_ptr + 1) / (448.0 * 448.0)
    else:
        qk_scale = sm_scale
        out_scale = 1.0

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
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
        # An index at or past the pool is invalid, exactly as this module's
        # contract states. Testing only `>= 0` let a stale slot -- the DSA
        # top-k writes into a buffer shared across layers, so an unwritten one
        # can hold any large positive value -- address past the pool's end.
        valid = (idx >= 0) & (idx < n_rows)
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


# Head tiling across the grid: {(cap): {H: HEAD_TILE}}.
#
# The kernel runs one program per query token with BLOCK_H covering every head.
# That is right while the `[BLOCK_H, d_v]` fp32 accumulator is small, and wrong
# once it is not: at H=64 it is 64*512*4 B over 256 threads = 128 of the
# kernel's 246 registers/thread, which pins the launch to 1 block/SM (NCU:
# Block Limit Registers 1, Block Limit Shared Mem 1, achieved occupancy 16.65%).
# Splitting the head dimension across the grid shrinks the accumulator at the
# cost of re-gathering each token's KV rows once per tile -- L2-resident, 77.6%
# hit, so the re-read is cheap relative to what the smaller accumulator buys.
#
# Conservative by construction, exactly like _PINNED_NARROW_H: an (arch, H) not
# listed here stays monolithic, so no untuned device or head count silently
# changes shape. Entries are added only from a measured sweep.
#
# Why this is a grid change and not a tile change: the whole 96-config
# (BLOCK_H, BLOCK_N, warps, stages) sweep recorded in this file holds
# BLOCK_H == H and varies the tile. Re-sweeping (BLOCK_N, warps, stages) at
# H=64 finds the pinned (16, 8, 2) already optimal at every T, so the tile space
# is exhausted; the grid shape is a different axis and had not been tried.
_PINNED_HEAD_TILE: dict = {
    # RTX 5080 / 5090 / PRO 6000 Blackwell, DeepSeek-V4 shape. Measured against
    # FlashInfer's SM120 route at T=512/1024/2048/4096; tile 16 wins at every T
    # for both head counts, tile 8 always loses (an 8x re-gather is too much) and
    # tile 32 does not compile to a usable shape here. H<=16 stays monolithic:
    # the tile would equal H and the grid dim collapses to 1.
    (12, 0): {32: 16, 64: 16},
    (12, 1): {32: 16, 64: 16},
}



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


def _head_tile(device, num_heads, mono, override=None):
    """Head rows per program.

    Returns ``mono`` -- the resolved ``_block_h``, i.e. one program per token
    with every head in it -- unless this (arch, head count) has a measured
    entry in ``_PINNED_HEAD_TILE``. An override of 0 or None means "use the
    table"; any other value is taken literally, which is what the sweep uses.

    The returned value is always a power of two and never exceeds ``mono``, so
    the default reproduces the untiled kernel exactly.
    """
    if override:
        tile = min(int(override), mono)
    else:
        cap = torch.cuda.get_device_capability(device)
        tile = _PINNED_HEAD_TILE.get(cap, {}).get(num_heads, mono)
    tile = max(1, min(triton.next_power_of_2(tile), mono))
    return tile


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
    head_tile=None,
    splits=1,
    partial_dtype=torch.bfloat16,
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
        splits: candidate-list splits per query token, for decode. ``1``
            (default) is the unsplit path this module has always run, bit for
            bit — no partials, no merge. ``"auto"`` picks a count from the SM
            count and the batch (see ``auto_splits``). Anything above 1 runs
            ``_nsa_decode_split_kernel`` on grid ``(T, splits)`` plus
            ``_nsa_decode_merge_kernel`` on grid ``(T, H)``; the result is
            algebraically identical but not bitwise, because the softmax is
            reassociated and the partials round to ``partial_dtype``.
        partial_dtype: element type of the ``[T, H, splits, d_v]`` partial
            output. bf16 matches FlashInfer; fp32 doubles the traffic and buys
            back the reassociation error.
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

    if splits == "auto":
        splits = auto_splits(q.device, T, h, topk, bn, block_h,
                             min_chunks=_MIN_CHUNKS_BF16)
    splits = int(splits)
    if splits > 1:
        # Decode split-K. Everything below this branch is untouched, so
        # `splits=1` keeps the base path bit for bit.
        mid_o, mid_m, mid_l = _split_ws(q.device, T, h, splits, d_v, partial_dtype)
        for bn_try, ns_try in _smem_fallbacks(bn, stages):
            try:
                _nsa_decode_split_kernel[(T, splits)](
                    q_in,
                    kv_in,
                    indices,
                    topk_length,
                    mid_o,
                    mid_m,
                    mid_l,
                    sm_scale,
                    topk,
                    kv_in.shape[0],
                    splits,
                    H=h,
                    BLOCK_H=block_h,
                    D_QK=d_qk,
                    D_V=d_v,
                    D_V_PAD=triton.next_power_of_2(d_v),
                    BLOCK_N=bn_try,
                    num_warps=warps,
                    num_stages=ns_try,
                    IDX64=idx64,
                )
                break
            except triton.runtime.errors.OutOfResources:
                continue
        else:
            raise triton.runtime.errors.OutOfResources(
                0, 0, "shared memory: no fallback config fits this device/shape"
            )
        _merge_launch(mid_o, mid_m, mid_l, out, attn_sink, T, h, d_v, splits)
        return out

    # Head tiling. `tile == block_h` is one program per token with every head in
    # it -- the untiled kernel, bitwise. Below that, grid dim 1 splits the head
    # dimension and the per-program accumulator shrinks with it; the tile's own
    # head count is then what picks (BLOCK_N, warps, stages), because that is
    # the shape the program actually runs.
    tile_h = _head_tile(q.device, h, block_h, head_tile)
    n_head_tiles = triton.cdiv(h, tile_h)
    if n_head_tiles > 1:
        bn, warps, stages = config or _config(q.device, tile_h)
    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        try:
            _nsa_prefill_kernel[(T, n_head_tiles)](
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
                kv_in.shape[0],
                H=h,
                BLOCK_H=tile_h,
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
    splits=1,
    partial_dtype=torch.bfloat16,
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

    if splits == "auto":
        splits = auto_splits(
            q.device, T, h, indices.shape[-1] + (x_topk if has_extra else 0),
            bn, block_h, min_chunks=_MIN_CHUNKS_FP8
        )
    splits = int(splits)
    if splits > 1:
        # Decode split-K over the concatenation of the two pools. See
        # `_nsa_decode_split_paged_fp8_kernel`. `splits=1` leaves the path below
        # bit for bit unchanged.
        mid_o, mid_m, mid_l = _split_ws(q.device, T, h, splits, D, partial_dtype)
        for bn_try, ns_try in _smem_fallbacks(bn, stages):
            try:
                _nsa_decode_split_paged_fp8_kernel[(T, splits)](
                    q, fp8_p, bf16_p, u8_p, indices, topk_length,
                    x_fp8, x_bf16, x_u8, extra_indices, extra_topk_length,
                    mid_o, mid_m, mid_l,
                    sm_scale,
                    indices.shape[-1],
                    x_topk,
                    splits,
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
                )
                break
            except triton.runtime.errors.OutOfResources:
                continue
        else:
            raise triton.runtime.errors.OutOfResources(
                0, 0, "shared memory: no fallback config fits this device/shape"
            )
        _merge_launch(mid_o, mid_m, mid_l, out, attn_sink, T, h, D, splits)
        return out

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


# ---------------------------------------------------------------------------
# Split-K decode (SM120). VARIANT ADDITION -- everything above this line is a
# byte-for-byte copy of
# sglang/python/sglang/kernels/ops/attention/dsa/triton_sparse_mla_prefill.py
# except for the two `splits=` arguments threaded into the two public entry
# points, which dispatch here and are a no-op at `splits <= 1`.
#
# Why this exists. The base path runs `grid = (T,)`, one program per query
# token. At decode T is the batch, so batch 1 occupies 1 SM of 84 and the wall
# time is flat (~0.039 ms bf16, ~0.168 ms paged fp8) from batch 1 to 64: the
# regime is latency/underfill, not work. Splitting a token's candidate list
# across S programs and merging the partials is the standard fix and is exactly
# what FlashInfer's own SM120 DSv4 decode kernel does
# (`sparse_mla_sm120_decode_dsv4.cu`): stage 1 on grid
# `(num_tokens, ceil(NH/16), num_splits)` writing `mid_out[T,H,splits,512]` bf16
# + `mid_lse[T,H,splits]` fp32, stage 2 merging on grid `(T, NH)`.
#
# Differences from FlashInfer's form, and why:
#
# * **Partials are unnormalised.** FlashInfer stores `acc/l` plus a single
#   `lse = log2(l) + m`; this stores raw `acc` plus `m` and `l` as two fp32
#   planes. `(acc, lse)` alone is not a sufficient statistic -- recovering the
#   merge weight `2^(m - gmax)` from `lse` needs `l` as well -- so the choice is
#   between one division per split at write time or one extra fp32 plane. The
#   plane is [T,H,S] (41 KB at T=128,H=8,S=10, against 82 KB of bf16 partials)
#   and keeps the split kernel's epilogue free of a divide.
# * **Natural-log domain, not log2.** The base path's `attn_sink` is a raw
#   natural-log logit and its online softmax runs on `tl.exp`; keeping the merge
#   in the same domain removes two LOG2E conversions and makes the S=1 merge
#   algebraically the same expression as the base path's epilogue.
# * **Whole-tile chunking.** A split covers `ceil(ceil(k_len/BLOCK_N)/S)` whole
#   BLOCK_N tiles, so every split starts on a tile boundary and gathers exactly
#   the addresses the unsplit loop would have. This is FlashInfer's
#   `chunks_per_block` with `chunks_per_block = ceil(num_chunks/S)`.
#
# The sink joins the denominator once, in the merge, never per split.
#
# Rejected before building (recorded so it is not retried): splitting over `D_V`
# instead of K. It needs no merge at all -- the splits write disjoint output
# slices -- but every split still re-gathers all 640 candidate rows, and the
# gather is ~50% of the runtime, so 4x the gather for 4x the parallelism is a
# wash.
#
# Measured (RTX 5080, 84 SMs, h=8, d=512, swa 128 + c4 512, attn_sink,
# probe/probe_splitk_decode.py, job 3594983). CUDA-graph replay, which is what
# an SGLang decode step pays; FlashInfer through SGLang's
# `_flash_mla_flashinfer` at heads=8 with its page split charged in:
#
#     B    fi h=8   bf16 base   bf16 split (S)   fi/split   fp8 base   fp8 split
#     1    0.0205     0.0391     0.0082 (10)      2.50x      0.1691    0.0143 (20)
#     4    0.0205     0.0389     0.0087 (10)      2.35x      0.1699    0.0164 (20)
#     16   0.0266     0.0410     0.0123 (10)      2.16x      0.1700    0.0471 (20)
#     64   0.0614     0.0411     0.0348 (10)      1.77x      0.1705    0.1556 (10)
#
# Two things that are not obvious from the older eager-mode numbers:
#
# * **The unsplit bf16 path loses to FlashInfer under CUDA graphs at B <= 32.**
#   Its 3.1x eager advantage was ~106 us of host overhead in FlashInfer's
#   wrapper, not GPU time: FlashInfer measures 0.124 ms eager and 0.0205 ms
#   captured, while this kernel measures 0.039 both ways. Split-K is what makes
#   the bf16 arm actually win, 1.77-2.50x, and the win is a real one.
# * **The paged-fp8 arm goes from 0.12x to 1.43x of FlashInfer at B=1** (0.169
#   -> 0.0143 ms, 11.8x its own unsplit form) and still loses above B=4. The
#   gather deficit is unchanged; splitting hides it at small batch and cannot at
#   large.
# ---------------------------------------------------------------------------

# Split-count policy. All three constants are measured on RTX 5080 (sm_120, 84
# SMs) at the DSv4 decode shape; see `auto_splits` for what each one does and
# `probe/probe_splitk_decode.py` section 4 for the surface they were fitted to.
#
# The tile count is what bounds the split count from above: DSv4's 640
# candidates at the pinned BLOCK_N=32 are 20 tiles, so 20 splits is the finest
# balanced split available (FlashInfer, chunking 64 at a time, reaches 10). The
# claim that going past 10 is "wasteful" holds for the bf16 arm and is false for
# the paged-fp8 one, which is 1.55x faster at S=20 than at S=10 for B <= 4.
_BLOCKS_PER_SM = 2  # NCU: the base tile is register- and smem-limited to 2/SM
_MAX_WAVES = 4  # past this the partial traffic costs more than the parallelism
_MIN_CHUNKS_BF16 = 2  # >= 64 candidates per split amortises a bf16 split
_MIN_CHUNKS_FP8 = 1  # the paged gather is ~4x costlier per candidate, so 32 does


@triton.jit
def _nsa_decode_split_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    mid_o_ptr,  # [T, H, splits, D_V] -- unnormalised acc
    mid_m_ptr,  # [T, H, splits] fp32 -- running max, -inf for an empty split
    mid_l_ptr,  # [T, H, splits] fp32 -- running denominator, 0 for an empty split
    sm_scale,
    topk,
    n_rows,  # rows in the KV pool; an index at or past it is invalid
    splits,  # runtime, not constexpr: one compilation serves every batch size
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    D_V_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX64: tl.constexpr,
):
    """`_nsa_prefill_kernel`'s loop over a slice of the candidate list.

    Identical body, different bounds, and no epilogue: no sink, no normalise.
    Program `(t, s)` owns candidate tiles `[s*C, (s+1)*C)` of token `t`.
    """
    t = tl.program_id(0)
    s = tl.program_id(1)
    D_TAIL: tl.constexpr = D_QK - D_V

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    dv = tl.arange(0, D_V_PAD)
    vmask = dv < D_V

    qb = q_ptr + t * H * D_QK
    q_main = tl.load(
        qb + h[:, None] * D_QK + dv[None, :],
        mask=hmask[:, None] & vmask[None, :],
        other=0.0,
    )
    if D_TAIL > 0:
        dt = tl.arange(0, D_TAIL)
        q_tail = tl.load(
            qb + h[:, None] * D_QK + (D_V + dt)[None, :], mask=hmask[:, None], other=0.0
        )

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V_PAD], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = tl.load(len_ptr + t)
    # Whole BLOCK_N tiles per split, so a split's gather addresses are exactly a
    # contiguous run of the ones the unsplit loop would have issued. An
    # over-allocated split gets lo >= hi and simply runs zero iterations, then
    # stores m = -inf / l = 0 / acc = 0, which the merge weights to nothing --
    # the same early-exit-with-a-sentinel that FlashInfer writes explicitly.
    c_tiles = tl.cdiv(tl.cdiv(k_len, BLOCK_N), splits)
    lo = s * c_tiles * BLOCK_N
    hi = tl.minimum(lo + c_tiles * BLOCK_N, k_len)
    for k0 in tl.range(lo, hi, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < hi, other=-1)
        # An index at or past the pool is invalid, exactly as this module's
        # contract states. Testing only `>= 0` let a stale slot -- the DSA
        # top-k writes into a buffer shared across layers, so an unwritten one
        # can hold any large positive value -- address past the pool's end.
        valid = (idx >= 0) & (idx < n_rows)
        if IDX64:
            row = tl.where(valid, idx, 0).to(tl.int64)
        else:
            row = tl.where(valid, idx, 0)
        kb = kv_ptr + row[:, None] * D_QK
        kv_main = tl.load(
            kb + dv[None, :], mask=valid[:, None] & vmask[None, :], other=0.0
        )

        qk = tl.dot(q_main, tl.trans(kv_main))
        if D_TAIL > 0:
            kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)
            qk = tl.dot(q_tail, tl.trans(kv_tail), qk)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    mbase = ((t * H + h).to(tl.int64) * splits) + s
    tl.store(mid_m_ptr + mbase, m_i, mask=hmask)
    tl.store(mid_l_ptr + mbase, l_i, mask=hmask)
    tl.store(
        mid_o_ptr + mbase[:, None] * D_V + dv[None, :],
        acc.to(mid_o_ptr.dtype.element_ty),
        mask=hmask[:, None] & vmask[None, :],
    )


@triton.jit
def _nsa_decode_split_paged_fp8_kernel(
    q_ptr,
    fp8_ptr,
    bf16_ptr,
    u8_ptr,
    idx_ptr,
    len_ptr,
    x_fp8_ptr,
    x_bf16_ptr,
    x_u8_ptr,
    x_idx_ptr,
    x_len_ptr,
    mid_o_ptr,
    mid_m_ptr,
    mid_l_ptr,
    sm_scale,
    topk,
    x_topk,
    splits,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
    D_NOPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
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
):
    """Split-K over the *concatenation* of the two pools.

    The two pools are one candidate list as far as the softmax is concerned, so
    the split axis is the concatenated tile index: tiles `[0, ta)` address the
    SWA pool and `[ta, ta+tb)` the compressed one. Splitting each pool
    separately would give the 128-wide sliding window and the 512-wide selection
    the same number of programs, which is 4x the wrong ratio.
    """
    t = tl.program_id(0)
    s = tl.program_id(1)

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    d = tl.arange(0, D)
    is_nope = d < D_NOPE
    rope_col = tl.where(is_nope, 0, d - D_NOPE)
    s_lane = tl.arange(0, D // SCALE_TILE)

    q = tl.load(
        q_ptr + t * H * D + h[:, None] * D + d[None, :], mask=hmask[:, None], other=0.0
    )

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D], tl.float32)
    n = tl.arange(0, BLOCK_N)

    k_len = tl.load(len_ptr + t)
    ta = tl.cdiv(k_len, BLOCK_N)
    if HAS_EXTRA:
        x_len = tl.load(x_len_ptr + t)
        tb = tl.cdiv(x_len, BLOCK_N)
    else:
        x_len = 0
        tb = 0
    c_tiles = tl.cdiv(ta + tb, splits)
    tlo = s * c_tiles
    thi = tl.minimum(tlo + c_tiles, ta + tb)

    a_lo = tl.minimum(tlo, ta) * BLOCK_N
    a_hi = tl.minimum(tl.minimum(thi, ta) * BLOCK_N, k_len)
    for k0 in tl.range(a_lo, a_hi, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < a_hi, other=-1)
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
        b_lo = tl.maximum(tlo - ta, 0) * BLOCK_N
        b_hi = tl.minimum(tl.maximum(thi - ta, 0) * BLOCK_N, x_len)
        for k0 in tl.range(b_lo, b_hi, BLOCK_N):
            idx = tl.load(
                x_idx_ptr + t * x_topk + k0 + n, mask=(k0 + n) < b_hi, other=-1
            )
            kv, valid = _paged_fp8_row_tile(
                x_fp8_ptr, x_bf16_ptr, x_u8_ptr, idx, d, is_nope, rope_col, s_lane,
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

    mbase = ((t * H + h).to(tl.int64) * splits) + s
    tl.store(mid_m_ptr + mbase, m_i, mask=hmask)
    tl.store(mid_l_ptr + mbase, l_i, mask=hmask)
    tl.store(
        mid_o_ptr + mbase[:, None] * D + d[None, :],
        acc.to(mid_o_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


@triton.jit
def _nsa_decode_merge_kernel(
    mid_o_ptr,
    mid_m_ptr,
    mid_l_ptr,
    o_ptr,
    sink_ptr,  # [H] fp32; unused when not HAS_SINK
    splits,
    H: tl.constexpr,
    D_V: tl.constexpr,
    D_V_PAD: tl.constexpr,
    SPLIT_PAD: tl.constexpr,  # next power of two >= splits; tl.arange needs one
    HAS_SINK: tl.constexpr,
):
    """One program per (token, head). Log-domain max over the splits, rescale,
    sum, fold the sink in once, divide once."""
    t = tl.program_id(0)
    hh = tl.program_id(1)

    sp = tl.arange(0, SPLIT_PAD)
    spm = sp < splits
    dv = tl.arange(0, D_V_PAD)
    vmask = dv < D_V

    base = (t * H + hh).to(tl.int64) * splits + sp
    m_all = tl.load(mid_m_ptr + base, mask=spm, other=-float("inf"))
    l_all = tl.load(mid_l_ptr + base, mask=spm, other=0.0)

    gm = tl.max(m_all, axis=0)
    if HAS_SINK:
        # The sink is a raw natural-log logit joining the denominator without a
        # value row; it enters the global max so exp(sink - gm) cannot overflow
        # when every real logit sits far below it. Exactly once, here -- the
        # split kernel never sees it.
        sink = tl.load(sink_ptr + hh).to(tl.float32)
        gm = tl.maximum(gm, sink)
    gm = tl.where(gm == -float("inf"), 0.0, gm)

    w = tl.where(m_all == -float("inf"), 0.0, tl.exp(m_all - gm))
    den = tl.sum(w * l_all, axis=0)
    if HAS_SINK:
        den += tl.exp(sink - gm)

    a = tl.load(
        mid_o_ptr + base[:, None] * D_V + dv[None, :],
        mask=spm[:, None] & vmask[None, :],
        other=0.0,
    )
    num = tl.sum(w[:, None] * a.to(tl.float32), axis=0)

    den = tl.where(den == 0.0, 1.0, den)
    tl.store(
        o_ptr + (t * H + hh).to(tl.int64) * D_V + dv,
        (num / den).to(o_ptr.dtype.element_ty),
        mask=vmask,
    )


def auto_splits(device, num_tokens, num_heads, topk_total, block_n, block_h,
                min_chunks=_MIN_CHUNKS_BF16, blocks_per_sm=_BLOCKS_PER_SM,
                max_waves=_MAX_WAVES):
    """How many candidate-list splits to run, host-side. Three rules, in order.

    1. **Splits are derived from a chunks-per-split, never the other way round.**
       ``S = ceil(n_tiles / cpb)`` is what the kernel actually computes, so
       choosing ``S`` directly can land on a ratio that leaves the last split
       short: at ``n_tiles=20``, ``S=3`` gives 7/7/6 tiles and measures *slower*
       than ``S=2``'s 10/10 (0.109 vs 0.091 ms, paged fp8, B=32). Enumerating
       ``cpb`` makes every split the same size by construction. This is
       FlashInfer's ``chunks_per_block`` parametrisation, and the reason for it.
    2. **Split until the device is full, then stop.** ``max_waves`` waves at
       ``blocks_per_sm`` resident blocks bounds the partial traffic, which is
       what makes over-splitting lose rather than merely stop helping: at B=64
       the paged-fp8 arm at S=20 writes and reads back 21 MB of partials and
       runs 0.176 ms against the unsplit path's 0.170 -- the only measured
       configuration where splitting is worse than not splitting.
    3. **``min_chunks`` is per-arm and measured, not a constant.** A split has a
       fixed cost (q load, epilogue, partial write) that its tiles amortise. The
       bf16 gather needs 2 tiles (64 candidates) to cover it; the paged-fp8
       gather is ~4x costlier per candidate, so 1 tile already does, and forcing
       2 on it costs 1.55x at B<=4 (0.0225 vs 0.0145 ms). Hence
       ``_MIN_CHUNKS_BF16 = 2``, ``_MIN_CHUNKS_FP8 = 1``.

    Rejected: FlashInfer's own rule (minimise ``ceil(waves) - waves`` among
    candidates within 3 integer waves). It optimises the tail gap of the last
    wave, which assumes waves serialise; on this shape they do not -- 160 blocks
    of 2 tiles beat 80 blocks of 4 tiles at B=16 on both arms, i.e. two nominal
    waves beat one. Its objective picks 5 splits at B=16 where 10 measures 1.17x
    faster.
    """
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    per_token = triton.cdiv(num_heads, block_h)  # blocks the base path already has
    n_tiles = triton.cdiv(topk_total, block_n)
    room = max(1, (max_waves * blocks_per_sm * sm) // max(1, num_tokens * per_token))
    s_cap = max(1, min(room, n_tiles // max(1, min_chunks)))
    cpb = max(min_chunks, triton.cdiv(n_tiles, s_cap))
    return max(1, triton.cdiv(n_tiles, cpb))


# Partial-output scratch, grown in place and keyed only by (device, dtype) so a
# server's decode loop allocates once. Reusing it also keeps the path
# CUDA-graph-capturable: nothing is allocated inside the captured region.
_SPLIT_WS = {}


def _split_ws(device, num_tokens, num_heads, splits, d_v, dtype):
    n_o = num_tokens * num_heads * splits * d_v
    n_s = num_tokens * num_heads * splits
    key = (device, dtype)
    ws = _SPLIT_WS.get(key)
    if ws is None or ws[0].numel() < n_o or ws[1].numel() < n_s:
        ws = (
            torch.empty(n_o, dtype=dtype, device=device),
            torch.empty(n_s, dtype=torch.float32, device=device),
            torch.empty(n_s, dtype=torch.float32, device=device),
        )
        _SPLIT_WS[key] = ws
    return ws


def _merge_launch(mid_o, mid_m, mid_l, out, attn_sink, T, h, d_v, splits):
    split_pad = triton.next_power_of_2(splits)
    _nsa_decode_merge_kernel[(T, h)](
        mid_o,
        mid_m,
        mid_l,
        out,
        attn_sink if attn_sink is not None else mid_l,  # unread placeholder
        splits,
        H=h,
        D_V=d_v,
        D_V_PAD=triton.next_power_of_2(d_v),
        SPLIT_PAD=split_pad,
        HAS_SINK=attn_sink is not None,
        num_warps=4 if split_pad <= 16 else 8,
        num_stages=1,
    )


# ===========================================================================
# EVERYTHING BELOW THIS LINE IS `variants/tsmp_paged_native.py`'s native
# paged-fp8 section, verbatim, plus one addition:
# `_nsa_decode_split_paged_fp8_native_kernel`, which is that section's
# `_native_tile` driven by the split/merge structure defined above.
#
# This file is therefore the union of two variants that were each measured on
# their own and never together:
#
#   A  variants/tsmp_paged_native.py -- the native gather. 6.21x the older
#      `sparse_mla_prefill_paged_fp8` at prefill (8.19 -> 1.32 ms, T=4096),
#      1.72x FlashInfer h=16, cos 0.9996209 vs an fp32 oracle over the
#      dequantised stored KV.
#   B  variants/tsmp_splitk.py -- the split-K decode. 1.77-2.50x FlashInfer on
#      bf16 under CUDA-graph replay, but its paged-fp8 arm was built on the
#      *old* gather and lost above B=4 (0.77x at 8, 0.39x at 64).
#
# Neither source file is modified; both are still on disk and still run.
#
# MEASURED (probe/probe_splitk_native.py, job 3595743; RTX 5080 sm_120, 84 SMs,
# h=8 d=512 swa 128 + c4 512, attn_sink, CUDA-graph replay -- which is what an
# SGLang decode step pays. FlashInfer is its own SM120 DSv4 decode kernel
# through SGLang's `_flash_mla_flashinfer` at heads=8, with the pbs 256 -> 64
# page split inside the timed call, i.e. charged to it):
#
#     B    fi h=8   native unsplit   native split-K (S)   vs fi   old-gather split
#     1    0.0205       0.0430          0.0103 (10)       2.00x     0.0144  1.43x
#     2    0.0205       0.0430          0.0103 (10)       2.00x     0.0155  1.32x
#     4    0.0205       0.0430          0.0103 (10)       2.00x     0.0162  1.26x
#     8    0.0205       0.0430          0.0102 (10)       2.00x     0.0252  0.81x
#     16   0.0250       0.0430          0.0123 (10)       2.03x     0.0471  0.53x
#     32   0.0392       0.0431          0.0184 (5)        2.13x     0.0912  0.43x
#     64   0.0611       0.0445          0.0348 (2)        1.75x     0.1555  0.39x
#     128  refused      0.0573          0.0573 (1)          --      0.2914    --
#
# **The crossover is gone.** The old-gather arm fell under 1.0x between B=4 and
# B=8 and reached 0.39x at 64; this one holds 1.75-2.13x across every batch size
# FlashInfer will run, so fp8 decode now clears the same >1.5x bar the bf16 arm
# and both prefill arms already cleared. FlashInfer refuses B=128 outright
# ("Unsupported sparse-MLA prefill configuration"), so that row has no baseline;
# `auto_splits` correctly returns S=1 there and the split path costs nothing.
#
# Accuracy, against an fp32 oracle over the DEQUANTISED STORED KV -- the only
# reference that means anything for a cache that stores fp8, since the original
# bf16 tensor is not what any reader of this cache can return:
#
#     cos 0.9995876 - 0.9996418 at every one of the 56 (B, S) pairs measured,
#     flat in S to the 6th decimal. The unsplit native kernel scores the same,
#     so splitting costs nothing here; the bf16 arm is 0.9999979.
#
# Split reassociation on its own, against this module's own unsplit output:
# cos >= 0.9999964, max_abs 0.00195 = exactly one bf16 ULP for outputs in
# [0.5, 1). `splits=1` is `torch.equal` to the unsplit kernel at every batch
# size and on three different tiles.
#
# The split kernel keeps the fp8 tensor core -- that was the whole point of A
# and splitting could have lost it. Compiled PTX, BLOCK_N=64:
#   _nsa_decode_split_paged_fp8_native_kernel  112 x m16n8k32...e4m3.e4m3.f32
#                                              +32 x m16n8k16...bf16 (the rope
#                                              dots), 48384 B shared
#   _nsa_decode_split_paged_fp8_kernel (old)     0 x e4m3, 128 x bf16, 90368 B
# ===========================================================================


# ---------------------------------------------------------------------------
# Native paged-fp8 sparse MLA (DeepSeek-V4, SM120).
#
# DSv4's KV cache is *already* fp8. Per token the pool holds
#
#     [0    : 448)  fp8 e4m3 "nope", under 7 ue8m0 scale bytes (one per 64)
#     [448  : 576)  bf16 "rope" tail, 64 elements
#     footer, 8 B   the 7 scale bytes (+1 pad), after the page's data region
#
# and the attention head is the whole 512 = 448 + 64, i.e. the rope half is part
# of both the key and the value. So the correct kernel *quantises nothing on the
# KV side*: it consumes the stored bytes and the stored exponents as they are,
# which is zero additional KV error by construction. Only Q is quantised.
#
# The previous attempt (`_paged_fp8_row_tile`, kept upstream) measured 2.33x
# *slower* than bf16 and emitted zero fp8 mma. The cause was structural, not
# tuning: it *computed* the KV tile in registers (dequantise -> scale ->
# `tl.where` select), and a register-resident value cannot be an mma operand, so
# Triton wrote it back to shared memory in two layouts -- 90368 B of smem
# against bf16's 41728, one block per SM instead of two, plus 256 scalar
# `st.shared.b16` and 56 `cvt`, and the same 64 bf16 `HMMA`.
#
# The fix is to never construct the KV tile:
#
#   * the 448-wide nope is loaded as 7 chunks of 64, each exactly one ue8m0
#     scale wide, and handed to `tl.dot` as fp8 with no conversion at all;
#   * the 64-wide bf16 rope tail gets its own 64-wide load and its own small
#     bf16 dot, instead of riding a full-width [BLOCK_N, 512] masked load;
#   * the scales are read at their true width (one byte per row per chunk) and
#     never broadcast to a [BLOCK_N, 512] fp32 tile.
#
# Every load is its natural width, so no load is column-masked and none is
# widened to a power of two.
#
# Why 7 chunks rather than one 448-wide dot: the ue8m0 scale varies along the
# reduction axis of QK, so it cannot be lifted out of a single dot. Chunking at
# the scale granularity makes it a per-chunk constant that folds into an fp32
# multiply on the [BLOCK_H, BLOCK_N] logit tile. This costs no mma: 7 dots of
# K=64 issue exactly the same number of m16n8k32 instructions as one dot of
# K=448 would.
#
# PV cannot use the same trick, because there the scale varies along the
# reduction axis (tokens) rather than across it -- which is also why the
# hardware block-scaled mma cannot express it, since that form needs the scale
# constant over 32 contiguous reduction elements. So PV follows FlashInfer
# (`sparse_mla_sm120/prefill_kernel.cuh` around 466-476 and 503-507): fold the
# KV dequant scale into P *before* quantising P, leaving the PV mma with no
# B-side scale and an epilogue of one fp32 multiply per accumulator element.
#
# MEASURED OUTCOME (RTX 5080, sm_120, T=4096, h=8, topk 128+512, best tile
# BLOCK_N=64 / warps=4 / stages=2 / BLOCK_H=8):
#
#   vs FlashInfer, 8 heads padded to 64 (what a TP8 deployment runs)   5.04x
#   vs FlashInfer at heads=16 (its narrowest instantiation, a bound)   1.72x
#   vs the previous paged-fp8 attempt                                  6.21x
#   vs this file's own bf16 path over a dequantised workspace          0.84x
#
# So it clears the >1.5x-over-FlashInfer bar but does NOT beat the bf16 path,
# and the reason is structural rather than tuning. Ablation at T=4096
# (probe_native_attrib.py), holding everything else fixed:
#
#   1 gather,  3 dots   0.331 ms
#   1 gather, 15 dots   0.464 ms    <- the 12 extra dots cost 0.133 ms
#   7 gathers, 15 dots  1.317 ms    <- the 6 extra gathers cost 0.853 ms
#
# The gather is ~65% of runtime and the dots the fp8 tensor core accelerates are
# ~10%. DSv4's per-64 ue8m0 layout forces seven chunk tiles to be live from the
# gather through both the QK dots and the PV dots, because Triton cannot slice a
# tensor: QK needs a per-chunk scale along the reduction axis, and PV needs a
# differently-scaled P per output chunk. Seven live tiles cost far more than the
# 2x on the mma saves. FlashInfer pays neither, because it hand-writes ldmatrix
# against an explicit smem layout with warp specialisation -- which is exactly
# the part Triton does not expose.
#
# Accuracy, against an fp32 oracle over the dequantised stored KV:
#   this kernel 0.9996209   (Q quantisation 72% of the 1-cos loss, P 28%)
#   bf16 path   0.9999979
# The KV side is exact by construction, so the whole gap is the price of
# reaching the fp8 tensor core: QK cannot take a bf16 Q against an fp8 K
# (Triton rejects the mixed dot outright -- "Unsupported rhs dtype fp8e4nv").
# ---------------------------------------------------------------------------

_FP8_DTYPE = torch.float8_e4m3fn

# Architectures whose tensor core executes e4m3 mma natively. sm_121 is left out
# deliberately: it software-emulates fp8 mma by upcasting, so `tl.dot` on fp8
# there costs *more* than the bf16 path it would replace. Note `_PINNED` does
# carry a (12, 1) entry, so this must be membership in an explicit set and never
# a `>=` comparison on the capability tuple.
_FP8_MMA_CAPS = frozenset({(8, 9), (12, 0)})


def _has_fp8_mma(device=None):
    """Whether this device runs e4m3 mma on the tensor core rather than by
    upcasting. See `_FP8_MMA_CAPS`."""
    return torch.cuda.get_device_capability(device) in _FP8_MMA_CAPS


@triton.jit
def _kv_chunk(
    fp8_ptr, u8_ptr, base, sbase, valid, col,
    I: tl.constexpr, SCALE_TILE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One 64-wide fp8 chunk and its ue8m0 dequant scale, both at true width.

    The tile is returned as fp8 and never converted, so it can be an mma operand
    in either orientation. The scale is one byte per row, not a [BLOCK_N, 512]
    broadcast.
    """
    kv = tl.load(
        fp8_ptr + base[:, None] + I * SCALE_TILE + col[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    # ue8m0 is a biased power-of-two exponent, so dequant is exp2, not a
    # multiply by a stored float. 255 is the non-finite encoding; clamping keeps
    # a corrupt footer from turning the whole tile into NaN.
    eb = tl.load(u8_ptr + sbase + I, mask=valid, other=0)
    e = eb.to(tl.float32)
    # tl.dot_scaled wants the B scale as [N, K//32]; one stored byte covers
    # SCALE_TILE=64 values, i.e. two 32-wide mma scale blocks.
    return (
        kv,
        tl.exp2(tl.minimum(e, 254.0) - 127.0),
        tl.broadcast_to(eb[:, None], (BLOCK_N, SCALE_TILE // 32)),
    )


@triton.jit
def _q_chunk(qb, hmask, col, sm_scale, I: tl.constexpr, SCALE_TILE: tl.constexpr):
    """Quantise one 64-wide Q chunk to fp8 with a per-(row, chunk) scale.

    Runs once per program, outside the k loop. Returning the scale with sm_scale
    and the fp8 range already folded in means the QK epilogue is a single
    multiply.
    """
    qc = tl.load(
        qb + I * SCALE_TILE + col[None, :], mask=hmask[:, None], other=0.0
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(qc), axis=1), 1e-30)
    return (qc * (448.0 / amax)[:, None]).to(tl.float8e4nv), amax * (sm_scale / 448.0)


@triton.jit
def _pv_chunk(acc, alpha, p, kv, ksc, PV_ROWMAX: tl.constexpr):
    """Accumulate one 64-wide output chunk of P @ V.

    The KV dequant scale varies along the reduction axis, so it cannot be lifted
    out of the dot and the hardware block-scaled mma cannot express it either
    (that form needs the scale constant over 32 contiguous reduction elements).
    Fold it into P before quantising P instead, exactly as FlashInfer does
    (`sparse_mla_sm120/prefill_kernel.cuh` 466-476, 503-507): the mma then takes
    no B-side scale and the epilogue is one fp32 multiply per element.

    p is in [0, 1] and ksc > 0, so ws >= 0 and the row max is the full range --
    no clamp is needed before the cast.
    """
    ws = p * ksc[None, :]
    if PV_ROWMAX:
        # Tightest scale, but a [BLOCK_H, BLOCK_N] reduction per chunk.
        amax = tl.maximum(tl.max(ws, axis=1), 1e-30)
        p8 = (ws * (448.0 / amax)[:, None]).to(tl.float8e4nv)
        return acc * alpha[:, None] + tl.dot(p8, kv) * (amax * (1.0 / 448.0))[:, None]
    # p is in [0, 1], so max_n(ksc) bounds every row of ws: the scale becomes a
    # per-chunk scalar and the reduction shrinks from [BLOCK_H, BLOCK_N] to
    # [BLOCK_N]. Costs mantissa only on rows whose own max is far below it.
    cmax = tl.maximum(tl.max(ksc, axis=0), 1e-30)
    p8 = (ws * (448.0 / cmax)).to(tl.float8e4nv)
    return acc * alpha[:, None] + tl.dot(p8, kv) * (cmax * (1.0 / 448.0))


@triton.jit
def _qk_chunk(qk, q8, qsc, kv, ksc, ksc_u8, QK_SCALED: tl.constexpr):
    """Accumulate one 64-wide chunk of Q @ K^T.

    QK_SCALED picks how the stored ue8m0 reaches the maths. Both forms issue the
    same number of m16n8k32 instructions:
      False -- plain fp8 mma, scale folded as an fp32 outer product on the
               [BLOCK_H, BLOCK_N] logit tile.
      True  -- `tl.dot_scaled`, which lowers on sm_120 to
               `mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X...ue8m0`,
               the instruction FlashInfer hand-writes, with zero dequant
               instructions. The Q scale stays outside because it is constant
               along the reduction axis, so lhs_scale is None.
    """
    if QK_SCALED:
        return qk + tl.dot_scaled(
            q8, None, "e4m3", tl.trans(kv), ksc_u8, "e4m3"
        ) * qsc[:, None]
    return qk + tl.dot(q8, tl.trans(kv)) * (qsc[:, None] * ksc[None, :])


@triton.jit
def _native_tile(
    # running online-softmax state, one accumulator per 64-wide output chunk
    m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar,
    # query: 7 fp8 nope chunks with their per-(row, chunk) dequant scales, and
    # the bf16 rope tail. Quantised once per program, outside the k loop.
    q0, q1, q2, q3, q4, q5, q6,
    s0, s1, s2, s3, s4, s5, s6,
    q_rope,
    # pool: three aliased views of the same bytes, plus this tile's token ids
    fp8_ptr, bf16_ptr, u8_ptr, idx,
    col, rcol, sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_NOPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BYTES_PER_PAGE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_BYTES_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    IDX64: tl.constexpr,
    QK_SCALED: tl.constexpr,
    PV_ROWMAX: tl.constexpr,
    ABL_LOAD: tl.constexpr,
    ABL_DOT: tl.constexpr,
):
    """One KV tile: gather, QK, online softmax, PV. Returns the updated state.

    Inlined into the caller and written as straight-line named values. Triton
    3.6 rejects Python list mutation inside `tl.static_range` ("kvs.append(kv)"
    fails to compile), and a list element rebound inside a dynamic `tl.range`
    would not be picked up as a loop-carried value, so the ten accumulators
    cross the k loop as plain names and the seven chunks are unrolled by hand.
    """
    valid = idx >= 0
    loc = tl.where(valid, idx, 0)
    if IDX64:
        loc = loc.to(tl.int64)
    page = loc // PAGE_SIZE
    in_page = loc % PAGE_SIZE
    # Byte offsets: the data region holds ROW_BYTES per token, and the page's
    # scale footer starts after it.
    base = page * BYTES_PER_PAGE + in_page * ROW_BYTES
    sbase = (
        page * BYTES_PER_PAGE + S_OFFSET_BYTES + in_page * SCALE_BYTES_PER_TOKEN
    )

    # ---- gather -----------------------------------------------------------
    # Explicitly unrolled: Triton 3.6 rejects Python list mutation inside
    # tl.static_range, so the seven chunks are named values. Nothing here is
    # converted or selected -- k0..k6 stay fp8 all the way into the mma, in both
    # orientations.
    k0, c0, u0 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 0, SCALE_TILE, BLOCK_N)
    if ABL_LOAD:
        # Attribution only, and deliberately wrong: chunk 0 stands in for all
        # seven, so every dot, reduction and quantisation still happens and the
        # only thing that changes is the gather instruction count, 7 -> 1. The
        # gap to the real kernel is what the seven separate gathers cost.
        k1, c1, u1 = k0, c0, u0
        k2, c2, u2 = k0, c0, u0
        k3, c3, u3 = k0, c0, u0
        k4, c4, u4 = k0, c0, u0
        k5, c5, u5 = k0, c0, u0
        k6, c6, u6 = k0, c0, u0
    else:
        k1, c1, u1 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 1, SCALE_TILE, BLOCK_N)
        k2, c2, u2 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 2, SCALE_TILE, BLOCK_N)
        k3, c3, u3 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 3, SCALE_TILE, BLOCK_N)
        k4, c4, u4 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 4, SCALE_TILE, BLOCK_N)
        k5, c5, u5 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 5, SCALE_TILE, BLOCK_N)
        k6, c6, u6 = _kv_chunk(fp8_ptr, u8_ptr, base, sbase, valid, col, 6, SCALE_TILE, BLOCK_N)
    # The rope tail gets its own 64-wide load rather than riding a full-width
    # [BLOCK_N, 512] masked one.
    rope = tl.load(
        bf16_ptr + (base[:, None] + D_NOPE) // 2 + rcol[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    # ---- QK: 7 fp8 chunk dots + one bf16 rope dot -------------------------
    # sm_scale is folded into each s_i for the fp8 chunks; the rope dot takes it
    # on the fp32 result rather than by pre-scaling q_rope, which would round a
    # second time in bf16.
    qk = tl.dot(q_rope, tl.trans(rope)) * sm_scale
    qk = _qk_chunk(qk, q0, s0, k0, c0, u0, QK_SCALED)
    if not ABL_DOT:
        qk = _qk_chunk(qk, q1, s1, k1, c1, u1, QK_SCALED)
        qk = _qk_chunk(qk, q2, s2, k2, c2, u2, QK_SCALED)
        qk = _qk_chunk(qk, q3, s3, k3, c3, u3, QK_SCALED)
        qk = _qk_chunk(qk, q4, s4, k4, c4, u4, QK_SCALED)
        qk = _qk_chunk(qk, q5, s5, k5, c5, u5, QK_SCALED)
        qk = _qk_chunk(qk, q6, s6, k6, c6, u6, QK_SCALED)
    qk = tl.where(valid[None, :], qk, -float("inf"))

    # ---- online softmax ---------------------------------------------------
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    # Guards a tile whose slots are all empty, where m_new is still -inf.
    m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
    alpha = tl.exp(m_i - m_safe)
    p = tl.exp(qk - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)

    # ---- PV ---------------------------------------------------------------
    a0 = _pv_chunk(a0, alpha, p, k0, c0, PV_ROWMAX)
    if not ABL_DOT:
        a1 = _pv_chunk(a1, alpha, p, k1, c1, PV_ROWMAX)
        a2 = _pv_chunk(a2, alpha, p, k2, c2, PV_ROWMAX)
        a3 = _pv_chunk(a3, alpha, p, k3, c3, PV_ROWMAX)
        a4 = _pv_chunk(a4, alpha, p, k4, c4, PV_ROWMAX)
        a5 = _pv_chunk(a5, alpha, p, k5, c5, PV_ROWMAX)
        a6 = _pv_chunk(a6, alpha, p, k6, c6, PV_ROWMAX)
    # The rope half of V is stored bf16; quantising it would add KV error the
    # cache does not already carry, so it keeps a bf16 dot.
    ar = ar * alpha[:, None] + tl.dot(p.to(tl.bfloat16), rope)

    return m_new, l_i, a0, a1, a2, a3, a4, a5, a6, ar


@triton.jit
def _nsa_prefill_paged_fp8_native_kernel(
    q_ptr,
    fp8_ptr,  # pool bytes viewed as float8_e4m3
    bf16_ptr,  # the same bytes viewed as bfloat16 (the rope tail)
    u8_ptr,  # the same bytes viewed as uint8 (the ue8m0 scale footer)
    idx_ptr,
    len_ptr,
    x_fp8_ptr,  # second pool, same three views; unused when not HAS_EXTRA
    x_bf16_ptr,
    x_u8_ptr,
    x_idx_ptr,
    x_len_ptr,
    o_ptr,
    sink_ptr,  # [H] fp32 learned per-head sink logit; unused when not HAS_SINK
    sm_scale,
    topk,
    x_topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_NOPE: tl.constexpr,  # fp8 half of a row (448)
    D_ROPE: tl.constexpr,  # bf16 tail (64); the head is D_NOPE + D_ROPE
    SCALE_TILE: tl.constexpr,  # fp8 values per ue8m0 byte (64)
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
    IDX64: tl.constexpr,
    QK_SCALED: tl.constexpr,
    PV_ROWMAX: tl.constexpr,
    ABL_LOAD: tl.constexpr,
    ABL_DOT: tl.constexpr,
):
    t = tl.program_id(0)
    D: tl.constexpr = D_NOPE + D_ROPE

    # Head tile, as in `_nsa_prefill_kernel`: grid dim 1 is ceil(H / BLOCK_H), so
    # a program owns one (token, head tile) rather than one token. Every use of
    # `h` below -- q, sink, output -- is already an ABSOLUTE head index, so
    # offsetting it here is the whole change. With BLOCK_H >= H the grid dim is
    # 1, `hb` is 0, and this is bit for bit the untiled kernel.
    hb = tl.program_id(1)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    col = tl.arange(0, SCALE_TILE)  # lanes within one fp8 chunk
    rcol = tl.arange(0, D_ROPE)  # lanes within the bf16 rope tail
    n = tl.arange(0, BLOCK_N)
    qb = q_ptr + t * H * D + h[:, None] * D

    # ---- Q, quantised once per program ------------------------------------
    # Per (row, chunk) amax: one reduction per chunk, outside the k loop, and
    # strictly tighter than a single row scale over all 448 dims. sm_scale and
    # the fp8 dequant both fold into the returned logit scale.
    q0, s0 = _q_chunk(qb, hmask, col, sm_scale, 0, SCALE_TILE)
    q1, s1 = _q_chunk(qb, hmask, col, sm_scale, 1, SCALE_TILE)
    q2, s2 = _q_chunk(qb, hmask, col, sm_scale, 2, SCALE_TILE)
    q3, s3 = _q_chunk(qb, hmask, col, sm_scale, 3, SCALE_TILE)
    q4, s4 = _q_chunk(qb, hmask, col, sm_scale, 4, SCALE_TILE)
    q5, s5 = _q_chunk(qb, hmask, col, sm_scale, 5, SCALE_TILE)
    q6, s6 = _q_chunk(qb, hmask, col, sm_scale, 6, SCALE_TILE)
    q_rope = tl.load(qb + D_NOPE + rcol[None, :], mask=hmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    a0 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a1 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a2 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a3 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a4 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a5 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a6 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    ar = tl.zeros([BLOCK_H, D_ROPE], tl.float32)

    k_len = tl.load(len_ptr + t)
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar = _native_tile(
            m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar,
            q0, q1, q2, q3, q4, q5, q6,
            s0, s1, s2, s3, s4, s5, s6,
            q_rope,
            fp8_ptr, bf16_ptr, u8_ptr, idx, col, rcol, sm_scale,
            BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, D_NOPE=D_NOPE,
            SCALE_TILE=SCALE_TILE, PAGE_SIZE=PAGE_SIZE,
            BYTES_PER_PAGE=BYTES_PER_PAGE, ROW_BYTES=ROW_BYTES,
            SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
            S_OFFSET_BYTES=S_OFFSET_BYTES, IDX64=IDX64,
            QK_SCALED=QK_SCALED, PV_ROWMAX=PV_ROWMAX,
            ABL_LOAD=ABL_LOAD, ABL_DOT=ABL_DOT,
        )

    if HAS_EXTRA:
        # Online softmax is associative over the concatenation, so the second
        # pool just continues the same running (m, l, acc). DSv4 passes the
        # sliding window here and the compressed cache in the first pool.
        x_len = tl.load(x_len_ptr + t)
        for k0 in tl.range(0, x_len, BLOCK_N):
            idx = tl.load(
                x_idx_ptr + t * x_topk + k0 + n, mask=(k0 + n) < x_len, other=-1
            )
            m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar = _native_tile(
                m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar,
                q0, q1, q2, q3, q4, q5, q6,
                s0, s1, s2, s3, s4, s5, s6,
                q_rope,
                x_fp8_ptr, x_bf16_ptr, x_u8_ptr, idx, col, rcol, sm_scale,
                BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, D_NOPE=D_NOPE,
                SCALE_TILE=SCALE_TILE, PAGE_SIZE=X_PAGE_SIZE,
                BYTES_PER_PAGE=X_BYTES_PER_PAGE, ROW_BYTES=ROW_BYTES,
                SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
                S_OFFSET_BYTES=X_S_OFFSET_BYTES, IDX64=IDX64,
                QK_SCALED=QK_SCALED, PV_ROWMAX=PV_ROWMAX,
            ABL_LOAD=ABL_LOAD, ABL_DOT=ABL_DOT,
            )

    if HAS_SINK:
        # A raw logit (already in natural-log space, not scaled by sm_scale)
        # that joins the denominator without contributing a value row. acc and
        # l_i are both held at m_base, so rescaling the pair to a combined max
        # leaves acc/l_i unchanged while keeping exp(sink - base) from
        # overflowing when every logit is far below sink.
        sink = tl.load(sink_ptr + h, mask=hmask, other=-float("inf")).to(tl.float32)
        m_base = tl.where(m_i == -float("inf"), 0.0, m_i)
        m_comb = tl.maximum(m_base, sink)
        rescale = tl.exp(m_base - m_comb)
        l_i = l_i * rescale + tl.exp(sink - m_comb)
        a0 = a0 * rescale[:, None]
        a1 = a1 * rescale[:, None]
        a2 = a2 * rescale[:, None]
        a3 = a3 * rescale[:, None]
        a4 = a4 * rescale[:, None]
        a5 = a5 * rescale[:, None]
        a6 = a6 * rescale[:, None]
        ar = ar * rescale[:, None]

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv = (1.0 / l_safe)[:, None]
    ob = o_ptr + t * H * D + h[:, None] * D
    ot = o_ptr.dtype.element_ty
    tl.store(ob + 0 * SCALE_TILE + col[None, :], (a0 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 1 * SCALE_TILE + col[None, :], (a1 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 2 * SCALE_TILE + col[None, :], (a2 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 3 * SCALE_TILE + col[None, :], (a3 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 4 * SCALE_TILE + col[None, :], (a4 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 5 * SCALE_TILE + col[None, :], (a5 * inv).to(ot), mask=hmask[:, None])
    tl.store(ob + 6 * SCALE_TILE + col[None, :], (a6 * inv).to(ot), mask=hmask[:, None])
    tl.store(
        ob + D_NOPE + rcol[None, :], (ar * inv).to(ot), mask=hmask[:, None]
    )


# ---------------------------------------------------------------------------
# Split-K decode over the *native* paged-fp8 gather. VARIANT ADDITION.
#
# This is the combination of the two pieces above: the split/merge structure of
# `_nsa_decode_split_kernel` + `_nsa_decode_merge_kernel` (grid (T,S) then
# (T,H), unnormalised partials as `acc` + two fp32 planes `m` and `l`, sink
# folded exactly once in the merge) applied to `_native_tile`, which reads the
# stored fp8 bytes and their ue8m0 exponents without converting anything.
#
# Why it had to be built. The split path's paged-fp8 arm was written against the
# *old* `_paged_fp8_row_tile` gather, which materialises the KV tile in
# registers and therefore emits no fp8 mma at all. On that base, splitting
# reached 1.43x of FlashInfer at B=1 but fell to 0.77x at B=8 and 0.39x at B=64:
# splitting hid the gather deficit at small batch and could not at large. The
# native gather is 6.21x faster at prefill on exactly the same bytes, so the
# question this kernel answers is whether the arm still crosses over, and where.
#
# Structural notes, all forced rather than chosen:
#
# * **The accumulator is eight named tiles, not one.** DSv4's ue8m0 layout puts
#   one scale per 64 nope values, and the PV scale varies along the reduction
#   axis, so P is quantised once per output chunk; the seven chunk accumulators
#   plus the bf16 rope accumulator therefore cross the k loop as separate
#   loop-carried values. The merge is unaffected: the eight tiles are stored to
#   byte offsets 0, 64, ..., 448 of the same 512-wide partial row, which is
#   exactly the contiguous `mid_o[t, h, s, :]` the merge already reads.
# * **The split range is over the concatenated tile index**, identical to
#   `_nsa_decode_split_paged_fp8_kernel`: tiles `[0, ta)` address the first pool
#   and `[ta, ta+tb)` the second, because the two pools are one candidate list
#   as far as the softmax is concerned.
# * **No sink and no normalise here.** Both belong to the merge, once, and that
#   is what makes S=1 dispatch the unsplit kernel rather than emulate it.
# ---------------------------------------------------------------------------


@triton.jit
def _nsa_decode_split_paged_fp8_native_kernel(
    q_ptr,
    fp8_ptr,  # pool bytes viewed as float8_e4m3
    bf16_ptr,  # the same bytes viewed as bfloat16 (the rope tail)
    u8_ptr,  # the same bytes viewed as uint8 (the ue8m0 scale footer)
    idx_ptr,
    len_ptr,
    x_fp8_ptr,  # second pool, same three views; unused when not HAS_EXTRA
    x_bf16_ptr,
    x_u8_ptr,
    x_idx_ptr,
    x_len_ptr,
    mid_o_ptr,  # [T, H, splits, D] -- unnormalised acc
    mid_m_ptr,  # [T, H, splits] fp32 -- running max, -inf for an empty split
    mid_l_ptr,  # [T, H, splits] fp32 -- running denominator, 0 when empty
    sm_scale,
    topk,
    x_topk,
    splits,  # runtime, not constexpr: one compilation serves every batch size
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
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
    IDX64: tl.constexpr,
    QK_SCALED: tl.constexpr,
    PV_ROWMAX: tl.constexpr,
):
    """`_nsa_prefill_paged_fp8_native_kernel`'s loop over a slice of the
    concatenated candidate list. Same body, different bounds, no epilogue."""
    t = tl.program_id(0)
    s = tl.program_id(1)
    D: tl.constexpr = D_NOPE + D_ROPE

    # Head tile on grid dim 2 -- dims 0 and 1 are already token and split. The
    # partials are addressed as `mid_o[t, h, s, :]` with an absolute `h`, so
    # tiling the heads leaves both the layout and `_merge_launch` untouched.
    hb = tl.program_id(2)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    col = tl.arange(0, SCALE_TILE)  # lanes within one fp8 chunk
    rcol = tl.arange(0, D_ROPE)  # lanes within the bf16 rope tail
    n = tl.arange(0, BLOCK_N)
    qb = q_ptr + t * H * D + h[:, None] * D

    # Q is quantised once per program, exactly as in the unsplit kernel. Every
    # split re-pays it; at DSv4's 7 chunks of 64 that is 7 reductions over a
    # [BLOCK_H, 64] tile, which is what `_MIN_CHUNKS_*` has to amortise.
    q0, s0 = _q_chunk(qb, hmask, col, sm_scale, 0, SCALE_TILE)
    q1, s1 = _q_chunk(qb, hmask, col, sm_scale, 1, SCALE_TILE)
    q2, s2 = _q_chunk(qb, hmask, col, sm_scale, 2, SCALE_TILE)
    q3, s3 = _q_chunk(qb, hmask, col, sm_scale, 3, SCALE_TILE)
    q4, s4 = _q_chunk(qb, hmask, col, sm_scale, 4, SCALE_TILE)
    q5, s5 = _q_chunk(qb, hmask, col, sm_scale, 5, SCALE_TILE)
    q6, s6 = _q_chunk(qb, hmask, col, sm_scale, 6, SCALE_TILE)
    q_rope = tl.load(qb + D_NOPE + rcol[None, :], mask=hmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    a0 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a1 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a2 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a3 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a4 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a5 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    a6 = tl.zeros([BLOCK_H, SCALE_TILE], tl.float32)
    ar = tl.zeros([BLOCK_H, D_ROPE], tl.float32)

    # Split over the concatenation of the two pools; see
    # `_nsa_decode_split_paged_fp8_kernel` for why the split axis cannot be
    # per-pool.
    k_len = tl.load(len_ptr + t)
    ta = tl.cdiv(k_len, BLOCK_N)
    if HAS_EXTRA:
        x_len = tl.load(x_len_ptr + t)
        tb = tl.cdiv(x_len, BLOCK_N)
    else:
        x_len = 0
        tb = 0
    c_tiles = tl.cdiv(ta + tb, splits)
    tlo = s * c_tiles
    thi = tl.minimum(tlo + c_tiles, ta + tb)

    a_lo = tl.minimum(tlo, ta) * BLOCK_N
    a_hi = tl.minimum(tl.minimum(thi, ta) * BLOCK_N, k_len)
    for k0 in tl.range(a_lo, a_hi, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < a_hi, other=-1)
        m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar = _native_tile(
            m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar,
            q0, q1, q2, q3, q4, q5, q6,
            s0, s1, s2, s3, s4, s5, s6,
            q_rope,
            fp8_ptr, bf16_ptr, u8_ptr, idx, col, rcol, sm_scale,
            BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, D_NOPE=D_NOPE,
            SCALE_TILE=SCALE_TILE, PAGE_SIZE=PAGE_SIZE,
            BYTES_PER_PAGE=BYTES_PER_PAGE, ROW_BYTES=ROW_BYTES,
            SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
            S_OFFSET_BYTES=S_OFFSET_BYTES, IDX64=IDX64,
            QK_SCALED=QK_SCALED, PV_ROWMAX=PV_ROWMAX,
            ABL_LOAD=False, ABL_DOT=False,
        )

    if HAS_EXTRA:
        b_lo = tl.maximum(tlo - ta, 0) * BLOCK_N
        b_hi = tl.minimum(tl.maximum(thi - ta, 0) * BLOCK_N, x_len)
        for k0 in tl.range(b_lo, b_hi, BLOCK_N):
            idx = tl.load(
                x_idx_ptr + t * x_topk + k0 + n, mask=(k0 + n) < b_hi, other=-1
            )
            m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar = _native_tile(
                m_i, l_i, a0, a1, a2, a3, a4, a5, a6, ar,
                q0, q1, q2, q3, q4, q5, q6,
                s0, s1, s2, s3, s4, s5, s6,
                q_rope,
                x_fp8_ptr, x_bf16_ptr, x_u8_ptr, idx, col, rcol, sm_scale,
                BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, D_NOPE=D_NOPE,
                SCALE_TILE=SCALE_TILE, PAGE_SIZE=X_PAGE_SIZE,
                BYTES_PER_PAGE=X_BYTES_PER_PAGE, ROW_BYTES=ROW_BYTES,
                SCALE_BYTES_PER_TOKEN=SCALE_BYTES_PER_TOKEN,
                S_OFFSET_BYTES=X_S_OFFSET_BYTES, IDX64=IDX64,
                QK_SCALED=QK_SCALED, PV_ROWMAX=PV_ROWMAX,
                ABL_LOAD=False, ABL_DOT=False,
            )

    # The eight accumulators tile the 512-wide partial row exactly, so the merge
    # reads them back as one contiguous `mid_o[t, h, s, :]` and needs no change.
    mbase = ((t * H + h).to(tl.int64) * splits) + s
    tl.store(mid_m_ptr + mbase, m_i, mask=hmask)
    tl.store(mid_l_ptr + mbase, l_i, mask=hmask)
    ob = mid_o_ptr + mbase[:, None] * D
    ot = mid_o_ptr.dtype.element_ty
    tl.store(ob + 0 * SCALE_TILE + col[None, :], a0.to(ot), mask=hmask[:, None])
    tl.store(ob + 1 * SCALE_TILE + col[None, :], a1.to(ot), mask=hmask[:, None])
    tl.store(ob + 2 * SCALE_TILE + col[None, :], a2.to(ot), mask=hmask[:, None])
    tl.store(ob + 3 * SCALE_TILE + col[None, :], a3.to(ot), mask=hmask[:, None])
    tl.store(ob + 4 * SCALE_TILE + col[None, :], a4.to(ot), mask=hmask[:, None])
    tl.store(ob + 5 * SCALE_TILE + col[None, :], a5.to(ot), mask=hmask[:, None])
    tl.store(ob + 6 * SCALE_TILE + col[None, :], a6.to(ot), mask=hmask[:, None])
    tl.store(ob + D_NOPE + rcol[None, :], ar.to(ot), mask=hmask[:, None])


# Split-count and tile policy for the native paged-fp8 arm. Everything this
# combination inherits was measured against a different kernel, so all of it was
# re-measured here (probe_splitk_native.py, job 3595646, RTX 5080 / sm_120 / 84
# SMs, CUDA-graph replay, DSv4 decode shape h=8 d=512 swa 128 + c4 512):
#
# * **The tile does carry over -- but that had to be checked.** A's
#   `(BLOCK_N, warps, stages, BLOCK_H)` was swept at *prefill*, T=4096, where the
#   grid alone fills the device; at decode BLOCK_N additionally fixes how finely
#   the candidate list can be split (640 candidates are 20 tiles at 32, 10 at
#   64, 5 at 128). Re-swept over 3x2x2 tiles at B in {1, 8, 64}, taking the best
#   S in each cell, (64, 4, 2) at BLOCK_H=8 wins again -- gmean 0.0151 ms
#   against 0.0154 for (32, 4, 2), 0.0168 for (64, 8, 2) and 0.0210 for
#   (128, 4, 2). The finer split BLOCK_N=32 buys (S up to 20) does not pay for
#   the doubled per-tile overhead. BLOCK_H 8 beats 16 only at B=64 (0.0328 vs
#   0.0349) and ties below.
#   Note this is NOT the module's `_config` tile: `_PINNED_NARROW_H` gives
#   (32, 4, 3) for h <= 8, which was swept for the *bf16* kernel and costs this
#   one 1.48x unsplit (0.0635 vs 0.0430 ms) -- so the native arm pins its own.
# * **`_MAX_WAVES` does not carry over: 4 -> 1.** It was fitted against the old
#   gather. Scoring the heuristic as the geometric mean over B of
#   `t[auto S] / t[best S]` on the dense S sweep: (min_chunks, max_waves) =
#   (1, 1) scores 1.008, (1, 3) 1.013, and B's own (1, 4) 1.033. A ~6x cheaper
#   gather makes a split's fixed cost relatively larger, so the point at which
#   more programs stop paying arrives a wave earlier.
# * **`_MIN_CHUNKS` does carry over at 1**, and the surface is steep about it:
#   min_chunks 2 scores 1.137, 3 scores 1.615, 4 scores 1.801. One BLOCK_N=64
#   tile per split still amortises the split's fixed cost (a 7-chunk Q
#   quantisation, an epilogue and a 512-wide partial write).
#
# Re-swept on sm_120 once head tiling landed, against the shipping
# (BLOCK_H=16, BLOCK_N=64, 4 warps). Six candidates, decode under CUDA-graph
# replay over B in {1..256} and prefill over T in {512..8192} (job 3626122,
# RTX 5080), as speedup vs shipping -- gmean decode / gmean prefill:
#
#     H16 N64 w8   0.78x / 0.61x     H32 N64 w8   0.91x / 0.86x
#     H8  N64 w8   0.57x / 0.36x     H16 N32 w8   0.58x / 0.39x
#     H64 N64 w4   0.45x / 0.35x  (the tile before head tiling)
#
# Nothing beats it, and the 8-warp arms lose worst -- which is the second time
# the static resource table has been wrong about speed in the same direction.
# It called (16, 64, 8) a strict improvement: zero spill against 1,320 B, twice
# the resident warps, identical shared memory, identical mma count. It measures
# 22% slower on decode and 39% on prefill, because eight warps widen the softmax
# row reduction into a deeper cross-warp tree and the barrier traffic costs more
# than the spill did. Shared memory and spill bytes are worth reading; they are
# not worth believing without a stopwatch.
_NATIVE_TILE = (64, 4, 2)  # (BLOCK_N, warps, stages); None -> `_config`
_NATIVE_BLOCK_H = 8  # floor for the head mma tile; raised to cover wider h
_MIN_CHUNKS_NATIVE = 1  # tiles per split below which a split does not pay
_MAX_WAVES_NATIVE = 1  # waves of `_BLOCKS_PER_SM` blocks before splitting loses

# What the last launch of each native arm actually ran, after `_smem_fallbacks`
# has had its say. Host-side only, so it costs nothing on the device and nothing
# inside a captured graph; it exists so that a tile sweep can tell a config that
# ran from one that silently stepped down to a smaller tile.
_LAUNCH_INFO = {}


def _native_config(device, num_heads):
    """The tile for the native fp8 arm, split or not.

    `_config` is the *bf16* kernel's pinned tile and is 1.48x slower here, so
    this arm carries its own. Falls back to `_config` on any device for which
    `_NATIVE_TILE` has not been swept.
    """
    bn, warps, stages = _NATIVE_TILE or _config(device, num_heads)
    # 8-bit operands need K >= 32 (`min_dot_size`), and BLOCK_N is the reduction
    # extent of the PV dot, so a narrower tile would not compile.
    return max(32, bn), warps, stages


# Head tile for the native arm, keyed the same way as `_PINNED_HEAD_TILE` but
# swept separately -- this kernel holds eight 64-wide fp32 accumulators per row
# rather than one [BLOCK_H, D_V] tile, so its register and smem curves are not
# the bf16 kernel's.
#
# At DeepSeek-V4's padded h=64 the untiled tile is BLOCK_H=64 at the pinned
# BLOCK_N=64, and on sm_120 that cubin **spills 9,104 B**: the eight 64-wide fp32
# accumulators overrun the 255-register cap, and they scale with BLOCK_H. Tiling
# to BLOCK_H=16 at the same BLOCK_N drops that to 1,320 B for an identical mma
# count, and measures **gmean 2.12x on decode** (1.75-2.88x over batch, CUDA-graph
# replay, RTX 5080, job 3624701).
#
# Occupancy is not the mechanism: both tiles are 1 block/SM and 4 warps/SM. The
# variant that *did* buy occupancy -- BLOCK_N narrowed to 32 for 3 blocks/SM --
# measured slower at every batch above 1. See `_PINNED_NATIVE_TILED_BN`.
#
# Resources read by compiling for sm_120 ahead of time (`metadata.shared` plus
# ptxas -v, probe/aot_sm120_native.py); the method and its traps are in
# probe/README_DSV4_SM120.md.
#
# BLOCK_H=8 is deliberately absent: mma's M granularity is 16, so an 8-row tile
# issues the same instructions for half the useful rows (2.00x the tensor work
# in the same sweep). It survives only where H itself is 8, where the tile
# cannot be wider than H anyway.
#
# No (12, 1) entry, unlike `_PINNED_HEAD_TILE`: sm_121 is not in
# `_FP8_MMA_CAPS`, so this arm never runs there, and a table row implying it was
# measured is the exact trap `test_fp8_gate_excludes_sm121` was written for.
_PINNED_NATIVE_HEAD_TILE: dict = {
    (12, 0): {32: 16, 64: 16},
}

# Narrowing BLOCK_N alongside the head tile was tried and **lost**; the table is
# kept empty so the negative result is not re-derived.
#
# The case for it was the shared-memory sweep: at BLOCK_H=16 on sm_120, BLOCK_N
# 32 compiles to 31,872 B (3 blocks/SM) against 54,528 B for 64 (1 block/SM).
# Measured under CUDA-graph replay at DSv4's decode shape it is slower at every
# batch above 1 -- 0.91x at B=2, 0.83x at 16, 0.78x at 32, 0.72x at 128 (job
# 3624701, RTX 5080). Head tiling alone at the inherited BLOCK_N=64 is gmean
# 2.12x over B against the untiled tile; adding this pin drops that to 1.82x.
#
# Occupancy on paper lost to the wider reduction tile, which is the same lesson
# the bf16 arm learned from NCU: on this kernel warps/SM is not the binding
# constraint that a shared-memory table makes it look like. Leaving it empty
# also keeps the tiled path on the *same* BLOCK_N as the untiled one, which is
# what makes the shipped default bitwise-comparable to it.
_PINNED_NATIVE_TILED_BN: dict = {}


def _native_head_tile(device, num_heads, mono, override=None):
    """Head rows per program for the native arm.

    Same contract as `_head_tile`: returns ``mono`` -- one program per token
    holding every head -- unless this (arch, head count) has a measured entry,
    and never exceeds ``mono``, so an unswept device reproduces the untiled
    kernel exactly.
    """
    if override:
        tile = min(int(override), mono)
    else:
        cap = torch.cuda.get_device_capability(device)
        tile = _PINNED_NATIVE_HEAD_TILE.get(cap, {}).get(num_heads, mono)
    return max(1, min(triton.next_power_of_2(tile), mono))


def _paged_fp8_layout(cache, page_size, d_nope, d_rope, scale_tile):
    """Three aliased flat views of one pool, plus its byte strides.

    Page interior is ``[P x row_bytes data][P x scale_bytes footer]``; a row is
    ``d_nope`` fp8 bytes then ``d_rope`` bf16 (DSv4: 448 + 128 = 576 B), and the
    per-token scale slot is padded to a power of two (7 scales in 8 B). Indices
    are absolute token ids into a contiguous pool -- there is no page table.
    """
    u8 = cache.view(torch.uint8)
    assert u8.is_contiguous(), "the paged cache must be contiguous"
    row_bytes = d_nope + d_rope * 2
    return (
        u8.view(_FP8_DTYPE).reshape(-1),
        u8.view(torch.bfloat16).reshape(-1),
        u8.reshape(-1),
        u8.shape[-1],  # bytes per page
        row_bytes,
        triton.next_power_of_2(d_nope // scale_tile),
        page_size * row_bytes,  # start of the page's scale footer
    )


def sparse_mla_prefill_paged_fp8_native(
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
    block_h=None,
    int64_indexing=None,
    qk_scaled=False,
    pv_rowmax=True,
    splits=1,
    partial_dtype=torch.bfloat16,
    _abl_load=False,
    _abl_dot=False,
):
    """Sparse-MLA prefill read straight off one or two paged fp8 KV pools.

    Consumes the stored fp8 bytes and the stored ue8m0 exponents directly, so
    the KV side contributes no quantisation error beyond what the cache already
    holds. Only Q is quantised (per row per 64-wide chunk).

    Args:
        q: ``[T, H, d_nope + d_rope]`` bf16.
        quant_k_cache: ``[num_pages, bytes_per_page]``, any dtype, read as bytes.
        indices: ``[T, topk]`` int32 absolute token ids; ``-1`` marks an empty
            slot.
        page_size: tokens per page of ``quant_k_cache``.
        extra_cache / extra_indices / extra_page_size: an optional second pool
            attended in the same softmax -- DSv4's compressed cache, which has
            its own page size.
        d_nope / d_rope: fp8 and bf16 halves of a row.
        scale_tile: fp8 values covered by one ue8m0 scale byte.
        attn_sink: optional ``[H]`` fp32 per-head sink logit.
        splits: candidate-list splits per query token, for decode. ``1``
            (default) runs ``_nsa_prefill_paged_fp8_native_kernel`` on grid
            ``(T,)``, bit for bit the kernel ``tsmp_paged_native.py`` ships on
            the same tile -- no partials, no merge.
            ``"auto"`` picks a count with ``auto_splits``
            under this arm's own ``_MIN_CHUNKS_NATIVE`` / ``_MAX_WAVES_NATIVE``.
            Anything above 1 runs
            ``_nsa_decode_split_paged_fp8_native_kernel`` on grid
            ``(T, splits)`` plus ``_nsa_decode_merge_kernel`` on ``(T, H)``;
            algebraically identical, not bitwise, because the softmax is
            reassociated and the partials round to ``partial_dtype``.
        partial_dtype: element type of the ``[T, H, splits, D]`` partial output.
    """
    if not _has_fp8_mma(q.device):
        cap = torch.cuda.get_device_capability(q.device)
        raise RuntimeError(
            f"sm_{cap[0]}{cap[1]} has no native e4m3 mma "
            f"(supported: {sorted(_FP8_MMA_CAPS)}). sm_121 in particular "
            "upcasts fp8, which makes this path slower than the bf16 one."
        )

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
    # The kernel carries one accumulator per scale chunk as a named value, so
    # the chunk count is fixed at DSv4's 7 rather than being a free parameter.
    if d_nope // scale_tile != 7:
        raise ValueError(
            f"this kernel is specialised to 7 scale chunks (DSv4's 448/64); got "
            f"{d_nope // scale_tile} from d_nope={d_nope}, "
            f"scale_tile={scale_tile}."
        )

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

    # One tile for both paths, so that `splits=1` and `splits>1` are the same
    # kernel body on the same tile and the only difference is the loop bounds
    # and the epilogue. `splits` is resolved after, because `auto_splits` needs
    # the tile the split kernel will actually run.
    if config is not None:
        # 8-bit operands need K >= 32 (`min_dot_size`), and BLOCK_N is the
        # reduction extent of the PV dot, so a narrower tile would not compile.
        bn, warps, stages = config
        bn = max(32, bn)
    else:
        bn, warps, stages = _native_config(q.device, h)
    # `_NATIVE_BLOCK_H` is a *floor* swept at DeepSeek-V4's post-TP8 h=8, not a
    # value: an untiled head mma tile must still cover every head, so a wider
    # head count raises it. Applying it unconditionally computed only the first
    # 8 rows and left the rest undefined (cos 0.58 at h=16). That resolved value
    # is now the *monolithic* tile, i.e. the ceiling `_native_head_tile` may cut
    # down to on a device where a narrower tile was measured.
    mono_h = max(_NATIVE_BLOCK_H or 0, triton.next_power_of_2(h))
    block_h = _native_head_tile(q.device, h, mono_h, override=block_h)
    if config is None and block_h < mono_h:
        # `_NATIVE_TILE`'s BLOCK_N was swept against the untiled tile; the head
        # tile changes which BLOCK_N fits. An explicit `config` still wins.
        bn = max(32, _PINNED_NATIVE_TILED_BN.get(
            torch.cuda.get_device_capability(q.device), bn))
    # Byte offsets, not element offsets: a pool is `num_pages * bytes_per_page`
    # bytes, so int32 addressing holds only for pools under 2 GiB.
    if int64_indexing is None:
        nbytes = max(
            quant_k_cache.numel() * quant_k_cache.element_size(),
            (extra_cache.numel() * extra_cache.element_size()) if has_extra else 0,
        )
        idx64 = nbytes > (2**31 - 1)
    else:
        idx64 = bool(int64_indexing)

    if splits == "auto":
        splits = auto_splits(
            q.device, T, h, indices.shape[-1] + (x_topk if has_extra else 0),
            bn, block_h, min_chunks=_MIN_CHUNKS_NATIVE,
            max_waves=_MAX_WAVES_NATIVE,
        )
    splits = int(splits)
    if splits > 1:
        # Decode split-K over the concatenation of the two pools. See
        # `_nsa_decode_split_paged_fp8_native_kernel`. `splits <= 1` leaves the
        # path below bit for bit unchanged.
        mid_o, mid_m, mid_l = _split_ws(q.device, T, h, splits, D, partial_dtype)
        for bn_try, ns_try in _smem_fallbacks(bn, stages):
            if bn_try < 32:
                continue
            try:
                _nsa_decode_split_paged_fp8_native_kernel[
                    (T, splits, triton.cdiv(h, block_h))
                ](
                    q, fp8_p, bf16_p, u8_p, indices, topk_length,
                    x_fp8, x_bf16, x_u8, extra_indices, extra_topk_length,
                    mid_o, mid_m, mid_l,
                    sm_scale,
                    indices.shape[-1],
                    x_topk,
                    splits,
                    H=h,
                    BLOCK_H=block_h,
                    D_NOPE=d_nope,
                    D_ROPE=d_rope,
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
                    IDX64=idx64,
                    QK_SCALED=bool(qk_scaled),
                    PV_ROWMAX=bool(pv_rowmax),
                )
                _LAUNCH_INFO["native_split"] = (bn_try, warps, ns_try, block_h)
                break
            except triton.runtime.errors.OutOfResources:
                continue
        else:
            raise triton.runtime.errors.OutOfResources(
                0, 0, "shared memory: no fallback config fits this device/shape"
            )
        _merge_launch(mid_o, mid_m, mid_l, out, attn_sink, T, h, D, splits)
        return out

    for bn_try, ns_try in _smem_fallbacks(bn, stages):
        if bn_try < 32:
            continue
        try:
            _nsa_prefill_paged_fp8_native_kernel[(T, triton.cdiv(h, block_h))](
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
                D_NOPE=d_nope,
                D_ROPE=d_rope,
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
                IDX64=idx64,
                QK_SCALED=bool(qk_scaled),
                PV_ROWMAX=bool(pv_rowmax),
                ABL_LOAD=bool(_abl_load),
                ABL_DOT=bool(_abl_dot),
            )
            _LAUNCH_INFO["native_unsplit"] = (bn_try, warps, ns_try, block_h)
            return out
        except triton.runtime.errors.OutOfResources:
            continue
    raise triton.runtime.errors.OutOfResources(
        0, 0, "shared memory: no fallback config fits this device/shape"
    )
