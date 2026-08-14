# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness for the fused Triton sparse-MLA prefill kernel.

Gates the base path and both opt-in exact fast paths against an fp32 reference
over the sparse-MLA contract. Each case guards a distinct failure mode:

- ``-1`` index padding and ragged rows (the indexer emits fewer than ``topk``
  selections for short prefixes).
- The union path's ownership mask: one gathered row set is shared by G query
  tokens, so a row selected by token A but not token B must be masked out of
  B's softmax. A mask bug here is invisible unless the shared set is genuinely
  larger than either token's own set, so the fixture builds overlapping-but-
  unequal sets rather than uniform-random ones.
- The dense-prefix identity: when ``t + 1 <= topk`` the selection is the whole
  causal prefix, and the kernel switches to a dense causal tile. Wrong guard =
  silently attending to the wrong rows.
- Head counts whose tuned tile exceeds the device shared-memory budget. Guards
  the h=32-on-SM120 launch failure (100 KB/CTA): the launcher must step the tile
  down, not propagate OutOfResources to the request.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    sparse_mla_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D_QK, D_V, SM_SCALE = 576, 512, 0.0625

# Widest head tile that still shares a layout with the narrower ones. Triton
# gives a [BLOCK_H, BLOCK_N] tile warpsPerCTA [1, 4] up to here and [4, 1] above
# it, which reverses the softmax row reduction from a cross-warp tree to an
# intra-warp one -- same arithmetic, different summation order, so results are
# bitwise comparable only within a family. Read off the TTGIR at BLOCK_H
# 8/16/32/64 for sm_120 (probe/aot_sm120_native.py).
_LAYOUT_FAMILY = 32


def _reference(q, kv, indices, d_v=D_V, attn_sink=None):
    """fp32 reference: each token attends over its own valid selected rows.

    ``attn_sink`` is DeepSeek-V4's learned per-head sink logit: a raw logit that
    joins the softmax denominator without contributing a value row.
    """
    T, h, _ = q.shape
    S = kv.shape[0]
    out = torch.empty(T, h, d_v, dtype=torch.float32, device=q.device)
    qf, kf = q.float(), kv.float()
    for t in range(T):
        idx = indices[t]
        idx = idx[(idx >= 0) & (idx < S)]
        if idx.numel() == 0:
            out[t] = 0.0
            continue
        k = kf[idx]
        logits = (qf[t] @ k.T) * SM_SCALE
        if attn_sink is None:
            p = torch.softmax(logits, dim=-1)
        else:
            sink = attn_sink.float()
            m = torch.maximum(logits.max(dim=-1).values, sink)
            e = torch.exp(logits - m[:, None])
            p = e / (e.sum(dim=-1) + torch.exp(sink - m))[:, None]
        out[t] = p @ k[:, :d_v]
    return out.to(torch.bfloat16)


def _qkv(T, S, h, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, h, D_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    kv = torch.randn(S, D_QK, dtype=torch.bfloat16, device="cuda", generator=g)
    return q, kv, g


def _random_indices(T, topk, S, g, pad_frac=0.0):
    idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
    n = max(1, int(topk * (1.0 - pad_frac)))
    for t in range(T):
        idx[t, :n] = torch.randperm(S, device="cuda", generator=g)[:n].to(torch.int32)
    return idx


def _overlapping_indices(T, topk, S, g):
    """Selections that mostly agree between neighbouring tokens, as the real
    indexer produces. Uniform-random sets are nearly disjoint, which makes the
    union tile degenerate to the per-token one and hides ownership-mask bugs.

    Rows stay unique within a token: top-k selection cannot pick a position
    twice, and the union path relies on that (it gathers the distinct union and
    masks per owner, so a repeated row would be weighted once instead of twice).
    """
    perm = torch.randperm(S, device="cuda", generator=g)
    pool, spare = perm[:topk], perm[topk:]
    n_keep = topk * 3 // 4
    n_fresh = topk - n_keep
    assert spare.numel() >= n_fresh, "S must exceed topk enough to vary the set"
    idx = torch.empty(T, topk, dtype=torch.int32, device="cuda")
    for t in range(T):
        keep = pool[torch.randperm(topk, device="cuda", generator=g)[:n_keep]]
        fresh = spare[
            torch.randperm(spare.numel(), device="cuda", generator=g)[:n_fresh]
        ]
        idx[t] = torch.cat([keep, fresh]).to(torch.int32)
    return idx


def _assert_matches(case, out, ref, tag, cos_min=0.999, max_abs=0.05):
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    mabs = (out.float() - ref.float()).abs().max().item()
    case.assertGreater(cos, cos_min, f"{tag}: cosine {cos:.6f}")
    case.assertLess(mabs, max_abs, f"{tag}: max_abs {mabs:.4f}")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDSATritonSparseMLAPrefill(CustomTestCase):
    def _assert_matches(self, out, ref, tag, **kw):
        _assert_matches(self, out, ref, tag, **kw)

    def test_base_path(self):
        for T, topk in ((512, 512), (2048, 2048), (37, 2048)):
            S = max(T, topk + 8)
            q, kv, g = _qkv(T, S, 8, seed=T)
            idx = _random_indices(T, topk, S, g)
            self._assert_matches(
                sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V),
                _reference(q, kv, idx),
                f"base T={T} topk={topk}",
            )

    def test_ragged_minus_one_padding(self):
        T, topk, S = 1024, 2048, 2056
        q, kv, g = _qkv(T, S, 8, seed=7)
        idx = _random_indices(T, topk, S, g, pad_frac=0.6)
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V),
            _reference(q, kv, idx),
            "ragged -1 padding",
        )

    def test_union_is_exact_on_overlapping_selections(self):
        # The union tile is only exercised when neighbouring tokens share rows;
        # each token must still see exactly its own set through the mask.
        for group in (2, 4):
            T, topk, S = (2048 // group) * group, 2048, 4096
            q, kv, g = _qkv(T, S, 8, seed=100 + group)
            idx = _overlapping_indices(T, topk, S, g)
            self._assert_matches(
                sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, union=group),
                _reference(q, kv, idx),
                f"union G={group}",
            )

    def test_dense_prefix_identity(self):
        T, topk = 2048, 2048
        q, kv, _ = _qkv(T, T, 8, seed=3)
        idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
        for t in range(T):
            n = min(t + 1, topk)
            idx[t, :n] = torch.arange(n, dtype=torch.int32, device="cuda")
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, dense=True),
            _reference(q, kv, idx),
            "dense-prefix",
        )

    def test_large_head_count_steps_down_instead_of_oom(self):
        T, topk, S = 256, 512, 520
        q, kv, g = _qkv(T, S, 32, seed=9)
        idx = _random_indices(T, topk, S, g)
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V),
            _reference(q, kv, idx),
            "h=32 smem fallback",
        )

    def test_union_large_head_count_steps_down(self):
        # The union Q tile is num_heads * G rows, so its shared-memory demand
        # grows faster than the per-token path's: 16 heads at G=2 overflows
        # SM120's 100 KB with the tuned tile. Guards the launch failure that
        # a TP4 deployment enabling union would otherwise hit.
        T, topk, S = 512, 512, 1024
        q, kv, g = _qkv(T, S, 16, seed=21)
        idx = _overlapping_indices(T, topk, S, g)
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, union=2),
            _reference(q, kv, idx),
            "union G=2 h=16 smem fallback",
        )

    def test_union_workspace_reuse_across_batch_shapes(self):
        # The union scratch is cached per (group size, span, device) and reused
        # across calls, with an epoch counter standing in for re-zeroing. Two
        # ways that goes wrong: a later, larger batch reading marks left by an
        # earlier one, and the epoch wrapping (every 128 calls) without clearing
        # rows the current batch does not cover. Alternate the batch size and
        # run past the wrap.
        topk, S = 512, 1024
        cases = []
        for T in (256, 1024, 256):
            q, kv, g = _qkv(T, S, 8, seed=T + 31)
            idx = _overlapping_indices(T, topk, S, g)
            cases.append((T, q, kv, idx, _reference(q, kv, idx)))

        for round_ in range(45):  # 45 * 3 calls > the 127-epoch wrap
            for T, q, kv, idx, ref in cases:
                out = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, union=2)
                if round_ in (0, 43, 44):
                    self._assert_matches(out, ref, f"union reuse T={T} round={round_}")

    def test_int64_indexing_matches_int32(self):
        # A KV pool past ~3.7M rows overflows int32 element offsets and the
        # launcher switches to int64 gather addressing. No test can allocate
        # that pool, so the mode is forced here instead: the two must agree
        # bitwise, or a large deployment silently reads the wrong rows.
        T, topk, S = 512, 512, 1024
        q, kv, g = _qkv(T, S, 8, seed=61)
        idx = _random_indices(T, topk, S, g)
        a = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, int64_indexing=False)
        b = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, int64_indexing=True)
        self.assertTrue(torch.equal(a, b), "int64 addressing changed the result")
        _assert_matches(self, b, _reference(q, kv, idx), "int64 indexing")

    def test_non_power_of_two_value_dim(self):
        # tl.arange cannot express 448, so the base path carries the value tile at
        # the next power of two and masks the surplus; the union and dense-prefix
        # paths index the value dim directly and must refuse rather than mis-tile.
        # (This is spare capacity, not DeepSeek-V4: sglang derives d_v = head_dim
        # = 512 for DSv4, which `test_no_rope_tail` covers instead.)
        T, topk, S, d_qk, d_v = 512, 512, 1024, 512, 448
        g = torch.Generator(device="cuda").manual_seed(71)
        q = torch.randn(T, 8, d_qk, dtype=torch.bfloat16, device="cuda", generator=g)
        kv = torch.randn(S, d_qk, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = _random_indices(T, topk, S, g)
        _assert_matches(
            self,
            sparse_mla_prefill(q, kv, idx, SM_SCALE, d_v),
            _reference(q, kv, idx, d_v),
            "d_v=448 base path",
        )
        for kwargs in ({"union": 2}, {"dense": True}):
            with self.assertRaisesRegex(ValueError, "power-of-two d_v"):
                sparse_mla_prefill(q, kv, idx, SM_SCALE, d_v, **kwargs)

    def test_no_rope_tail(self):
        # DeepSeek-V4 does not hand the kernel a separate rope tail: sglang
        # derives v_head_dim = head_dim = 512 for DeepseekV4ForCausalLM, and the
        # dequantized KV workspace is DIM_NOPE 448 + DIM_ROPE 64 = 512 wide, so
        # the 512-wide head is the key AND the whole value. d_qk == d_v makes
        # tl.arange(0, d_qk - d_v) a compile error, so the tail dot has to be
        # elided at trace time rather than masked.
        T, topk, S, d = 512, 640, 4096, 512  # topk 640 = align(512 + swa 128, 128)
        for h in (8, 16, 64):
            with self.subTest(h=h):
                g = torch.Generator(device="cuda").manual_seed(80 + h)
                q = torch.randn(T, h, d, dtype=torch.bfloat16, device="cuda", generator=g)
                kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
                idx = _random_indices(T, topk, S, g)
                self._assert_matches(
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, d),
                    _reference(q, kv, idx, d),
                    f"d_qk == d_v == {d}, h={h}",
                )

    def test_no_rope_tail_union(self):
        # The union tile is where the DSv4 win lives: its combined index set is
        # SWA(128) + top-k(512), and the SWA half shifts by exactly one position
        # per query token, so G neighbours share 128 + (G-1) of those rows by
        # construction. Same zero-width-tail elision as the base path.
        T, topk, S, d = 2048, 640, 4096, 512
        for group in (2, 4):
            with self.subTest(group=group):
                g = torch.Generator(device="cuda").manual_seed(82 + group)
                q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
                kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
                idx = _overlapping_indices(T, topk, S, g)
                self._assert_matches(
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, d, union=group),
                    _reference(q, kv, idx, d),
                    f"union G={group}, tail=0",
                )

    def test_no_rope_tail_dense_prefix(self):
        # Dense-prefix at DSv4's shape. Its payoff is smaller here than on GLM
        # (topk 640 vs 2048 means fewer tokens cover their whole prefix), but the
        # exact-set guard makes enabling it safe either way.
        T, topk, d = 1024, 1024, 512
        g = torch.Generator(device="cuda").manual_seed(84)
        q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
        kv = torch.randn(T, d, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
        for t in range(T):
            n = min(t + 1, topk)
            idx[t, :n] = torch.arange(n, dtype=torch.int32, device="cuda")
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, d, dense=True),
            _reference(q, kv, idx, d),
            "dense-prefix, tail=0",
        )

    def test_attn_sink(self):
        # DeepSeek-V4 carries a learned per-head sink logit into every softmax
        # (layers.N.attn.attn_sink, fp32[num_heads]); the real checkpoint's
        # values are O(1), so dropping it is a silent accuracy bug rather than a
        # rounding difference. Semantics match `_apply_attn_sink` in the SM120
        # decode path: logaddexp(lse, sink), no sm_scale applied to the sink.
        T, topk, S, d = 512, 640, 4096, 512
        for h in (8, 16):
            with self.subTest(h=h):
                g = torch.Generator(device="cuda").manual_seed(90 + h)
                q = torch.randn(T, h, d, dtype=torch.bfloat16, device="cuda", generator=g)
                kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
                idx = _random_indices(T, topk, S, g)
                sink = torch.randn(h, dtype=torch.float32, device="cuda", generator=g)
                self._assert_matches(
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, d, attn_sink=sink),
                    _reference(q, kv, idx, d, attn_sink=sink),
                    f"attn_sink h={h}",
                )

    def test_attn_sink_dominates_empty_selection(self):
        # A row with no valid selection has no value mass at all, so the sink
        # takes the whole softmax and the output must stay exactly zero rather
        # than divide by an empty denominator.
        T, topk, S, d = 64, 128, 256, 512
        g = torch.Generator(device="cuda").manual_seed(91)
        q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
        kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        out = sparse_mla_prefill(q, kv, idx, SM_SCALE, d, attn_sink=sink)
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_attn_sink_on_union(self):
        # DSv4 always carries a sink, so a fast path that cannot take one is a
        # fast path DSv4 cannot use. Union rows are (token, head) laid out
        # head-fastest -- a wrong sink lane here mixes heads and still looks
        # roughly right, so check it against the per-head reference.
        T, topk, S, d = 2048, 640, 4096, 512
        for group in (2, 4):
            with self.subTest(group=group):
                g = torch.Generator(device="cuda").manual_seed(93 + group)
                q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
                kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
                idx = _overlapping_indices(T, topk, S, g)
                sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
                self._assert_matches(
                    sparse_mla_prefill(
                        q, kv, idx, SM_SCALE, d, attn_sink=sink, union=group
                    ),
                    _reference(q, kv, idx, d, attn_sink=sink),
                    f"union G={group} + attn_sink",
                )

    def test_attn_sink_on_dense_prefix(self):
        T, topk, d = 1024, 1024, 512
        g = torch.Generator(device="cuda").manual_seed(95)
        q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
        kv = torch.randn(T, d, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = torch.full((T, topk), -1, dtype=torch.int32, device="cuda")
        for t in range(T):
            n = min(t + 1, topk)
            idx[t, :n] = torch.arange(n, dtype=torch.int32, device="cuda")
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, d, attn_sink=sink, dense=True),
            _reference(q, kv, idx, d, attn_sink=sink),
            "dense-prefix + attn_sink",
        )

    def test_attn_sink_on_glm_shape_fast_paths(self):
        # The sink must also survive the tail>0 form of both fast paths, so the
        # two shapes do not diverge into separately-maintained code.
        T, topk, S = 2048, 2048, 4096
        q, kv, g = _qkv(T, S, 8, seed=96)
        idx = _overlapping_indices(T, topk, S, g)
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        self._assert_matches(
            sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, attn_sink=sink, union=4),
            _reference(q, kv, idx, D_V, attn_sink=sink),
            "GLM shape, union=4 + attn_sink",
        )

    def test_deterministic(self):
        # No split-K / atomics / partial merge, so repeated calls must be
        # bitwise identical -- the property that lets a served model be
        # reproduced run to run.
        T, topk, S = 1024, 2048, 2056
        q, kv, g = _qkv(T, S, 8, seed=11)
        idx = _random_indices(T, topk, S, g)
        a = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V)
        b = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V)
        self.assertTrue(torch.equal(a, b), "kernel output is not deterministic")


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestDSATritonPrefillBackendAdapter(CustomTestCase):
    """The backend method itself, driving the real kernel.

    The CPU unit test covers this method with the kernel mocked, which pins the
    argument marshalling but cannot catch a wrong result or a wrong output
    contract. This runs it for real and checks both: the values against an fp32
    reference, and the shape against what the sibling `_forward_flashmla_sparse`
    returns to the same caller — `[num_tokens, num_heads, v_head_dim]`. Returning
    a different rank here would corrupt every downstream projection.
    """

    def _backend(self, *, union=0, dense=False):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_triton_union = union
        backend.dsa_triton_dense_prefix = dense
        return backend

    def test_forward_matches_reference_and_output_contract(self):
        T, topk, S, h = 1024, 512, 2048, 8
        q, kv, g = _qkv(T, S, h, seed=51)
        idx = _overlapping_indices(T, topk, S, g)
        ref = _reference(q, kv, idx)

        for union, dense in ((0, False), (4, False), (0, True)):
            with self.subTest(union=union, dense=dense):
                out = self._backend(
                    union=union, dense=dense
                )._forward_triton_sparse_mla(
                    q_all=q,
                    kv_cache=kv,
                    page_table_1=idx,
                    sm_scale=SM_SCALE,
                    v_head_dim=D_V,
                )
                self.assertEqual(
                    tuple(out.shape),
                    (T, h, D_V),
                    "must match the [num_tokens, num_heads, v_head_dim] that "
                    "_forward_flashmla_sparse returns to the same caller",
                )
                self.assertEqual(out.dtype, torch.bfloat16)
                _assert_matches(self, out, ref, f"backend union={union} dense={dense}")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Paged fp8 gather. The oracle is an independent torch decode of the same bytes
# rather than SGLang's Triton dequantizer, so a shared bug in the byte-layout
# arithmetic cannot make both sides agree.
# ---------------------------------------------------------------------------

D_NOPE, D_ROPE, SCALE_TILE = 448, 64, 64
ROW_BYTES = D_NOPE + D_ROPE * 2  # 576
SCALE_PER_TOK = 8  # 7 ue8m0 + 1 pad


def _bytes_per_page(page_size):
    per = ROW_BYTES + SCALE_PER_TOK
    return -(-page_size * per // ROW_BYTES) * ROW_BYTES


def _build_paged_fp8_pool(n_tokens, page_size, seed):
    """A DSv4-layout pool plus the bf16 rows it must decode to.

    Quantization mirrors the pool writer: one ue8m0 exponent per SCALE_TILE nope
    values chosen so the tile's amax lands inside e4m3 range; rope stored as bf16.
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    n_pages = -(-n_tokens // page_size)
    bpp = _bytes_per_page(page_size)
    pool = torch.zeros(n_pages, bpp, dtype=torch.uint8, device=dev)

    nope = torch.randn(n_tokens, D_NOPE, dtype=torch.float32, device=dev, generator=g)
    rope = torch.randn(n_tokens, D_ROPE, dtype=torch.bfloat16, device=dev, generator=g)
    tiles = nope.view(n_tokens, D_NOPE // SCALE_TILE, SCALE_TILE)
    exp = torch.ceil(
        torch.log2(tiles.abs().amax(dim=-1).clamp(min=1e-30) / 448.0)
    ).clamp(-127, 128)
    q_fp8 = (tiles / torch.exp2(exp)[:, :, None]).to(torch.float8_e4m3fn)
    ue8m0 = (exp + 127).to(torch.uint8)

    flat = pool.view(-1)
    tok = torch.arange(n_tokens, device=dev)
    data = (tok // page_size) * bpp + (tok % page_size) * ROW_BYTES
    scal = (tok // page_size) * bpp + page_size * ROW_BYTES + (
        tok % page_size
    ) * SCALE_PER_TOK
    flat[(data[:, None] + torch.arange(D_NOPE, device=dev)[None, :]).reshape(-1)] = (
        q_fp8.reshape(n_tokens, D_NOPE).view(torch.uint8).reshape(-1)
    )
    flat[
        (data[:, None] + (torch.arange(D_ROPE * 2, device=dev) + D_NOPE)[None, :])
        .reshape(-1)
    ] = rope.view(torch.uint8).reshape(-1)
    flat[
        (scal[:, None] + torch.arange(D_NOPE // SCALE_TILE, device=dev)[None, :])
        .reshape(-1)
    ] = ue8m0.reshape(-1)

    # Independent oracle: what the bytes decode to.
    decoded_nope = (
        q_fp8.to(torch.float32) * torch.exp2(exp)[:, :, None]
    ).reshape(n_tokens, D_NOPE)
    kv = torch.cat([decoded_nope.to(torch.bfloat16), rope], dim=1)
    return pool, kv


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestPagedFP8SparseMLAPrefill(CustomTestCase):
    """The gather-from-the-pool variant, against the bf16 gather of the same rows."""

    def _check(self, got, ref, tag):
        _assert_matches(self, got, ref, tag, cos_min=0.9999, max_abs=0.02)

    def test_matches_bf16_gather(self):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            sparse_mla_prefill_paged_fp8,
        )

        S, PAGE, T, topk = 2048, 256, 512, 640
        pool, kv = _build_paged_fp8_pool(S, PAGE, seed=11)
        for h in (8, 16, 64):  # 64 is what DSv4 pads to today
            with self.subTest(h=h):
                g = torch.Generator(device="cuda").manual_seed(h)
                q = torch.randn(
                    T, h, 512, dtype=torch.bfloat16, device="cuda", generator=g
                )
                idx = _random_indices(T, topk, S, g)
                sink = torch.randn(h, dtype=torch.float32, device="cuda", generator=g)
                self._check(
                    sparse_mla_prefill_paged_fp8(
                        q, pool, idx, SM_SCALE, PAGE, attn_sink=sink
                    ),
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, 512, attn_sink=sink),
                    f"paged fp8 h={h}",
                )

    def test_two_pools_share_one_softmax(self):
        # A DSv4 compressed layer attends the sliding window (SWA pool, page 256)
        # and the selected compressed rows (its own pool, page 64) in one softmax.
        # Concatenating them first is the materialization this variant avoids, so
        # the two-source path must equal the concatenated single-source one.
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            sparse_mla_prefill_paged_fp8,
        )

        S, XS, T, h = 2048, 1024, 512, 8
        pool, kv = _build_paged_fp8_pool(S, 256, seed=21)
        xpool, xkv = _build_paged_fp8_pool(XS, 64, seed=22)
        g = torch.Generator(device="cuda").manual_seed(23)
        q = torch.randn(T, h, 512, dtype=torch.bfloat16, device="cuda", generator=g)
        sidx = _random_indices(T, 128, S, g)
        xidx = _random_indices(T, 512, XS, g)
        sink = torch.randn(h, dtype=torch.float32, device="cuda", generator=g)
        ref = sparse_mla_prefill(
            q,
            torch.cat([kv, xkv], dim=0),
            torch.cat([sidx, torch.where(xidx >= 0, xidx + S, xidx)], dim=1),
            SM_SCALE,
            512,
            attn_sink=sink,
        )
        self._check(
            sparse_mla_prefill_paged_fp8(
                q, pool, sidx, SM_SCALE, 256, extra_cache=xpool,
                extra_indices=xidx, extra_page_size=64, attn_sink=sink,
            ),
            ref,
            "SWA pool + compressed pool",
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestSplitKDecode(CustomTestCase):
    """Split-K decode. The property that matters is that it changes nothing
    unless asked: ``splits=1`` must be the existing kernel, bit for bit."""

    def test_splits_one_is_bitwise_identical(self):
        S, topk = 4096, 640
        for B in (1, 8, 64):
            with self.subTest(B=B):
                q, kv, g = _qkv(B, S, 8, seed=B)
                idx = _random_indices(B, topk, S, g)
                sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
                base = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, attn_sink=sink)
                one = sparse_mla_prefill(
                    q, kv, idx, SM_SCALE, D_V, attn_sink=sink, splits=1
                )
                self.assertTrue(
                    torch.equal(base, one), f"splits=1 diverged from default at B={B}"
                )

    def test_split_matches_unsplit(self):
        # Splitting reassociates the softmax, so the difference is reduction
        # order alone -- one bf16 ulp for outputs in [0.5, 1).
        S, topk = 4096, 640
        for B in (1, 8, 64):
            for splits in (2, 4, 10):
                with self.subTest(B=B, splits=splits):
                    q, kv, g = _qkv(B, S, 8, seed=B + splits)
                    idx = _random_indices(B, topk, S, g)
                    sink = torch.randn(
                        8, dtype=torch.float32, device="cuda", generator=g
                    )
                    ref = sparse_mla_prefill(
                        q, kv, idx, SM_SCALE, D_V, attn_sink=sink
                    )
                    got = sparse_mla_prefill(
                        q, kv, idx, SM_SCALE, D_V, attn_sink=sink, splits=splits
                    )
                    _assert_matches(
                        self, got, ref, f"split B={B} S={splits}",
                        # Same story as `test_matches_bf16_gather`: 0.01 held on
                        # a 5080/5090 and an RTX PRO 6000 measures 0.015625 at
                        # (B=1, S=4), byte-identical whether the merge runs at 4
                        # warps or 2. cos stays at 0.9999969.
                        cos_min=0.99999, max_abs=0.02,
                    )

    def test_split_auto_is_correct(self):
        S, topk = 4096, 640
        for B in (1, 16, 128):
            with self.subTest(B=B):
                q, kv, g = _qkv(B, S, 8, seed=B * 3)
                idx = _random_indices(B, topk, S, g)
                sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
                _assert_matches(
                    self,
                    sparse_mla_prefill(
                        q, kv, idx, SM_SCALE, D_V, attn_sink=sink, splits="auto"
                    ),
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, attn_sink=sink),
                    f"auto-split B={B}",
                    cos_min=0.99999,
                    max_abs=0.01,
                )

    def test_merge_warps_follows_split_pad(self):
        # The merge's warp count comes from the height of the tile it reduces,
        # not from the batch. This guards the shape of that rule rather than its
        # speed: the numbers behind it were measured on two parts, but a later
        # edit that inverted it, or that returned 1 for a tall tile, would not
        # fail any accuracy test here.
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _merge_warps,
        )

        for split_pad, want in ((1, 2), (2, 2), (4, 2), (8, 1), (16, 1), (32, 8)):
            with self.subTest(split_pad=split_pad):
                self.assertEqual(_merge_warps(split_pad), want)

    def test_merge_warp_count_does_not_move_the_answer(self):
        # 1 warp is live at SPLIT_PAD >= 8, and the merge reduces along the
        # splits across threads, so fewer threads means a different reduction
        # tree. That is allowed to move the last bits; it is not allowed to move
        # the answer.
        import sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill as _k

        S, topk, B = 4096, 640, 8
        q, kv, g = _qkv(B, S, 8, seed=77)
        idx = _random_indices(B, topk, S, g)
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        prev = _k._MERGE_WARPS
        try:
            out = {}
            for w in (1, 2, 4):
                _k._MERGE_WARPS = w
                out[w] = sparse_mla_prefill(
                    q, kv, idx, SM_SCALE, D_V, attn_sink=sink, splits=10
                )
            for w in (1, 4):
                _assert_matches(
                    self, out[w], out[2], f"merge warps {w} vs 2",
                    cos_min=0.99999, max_abs=0.02,
                )
        finally:
            _k._MERGE_WARPS = prev

    def test_ragged_lengths_survive_splitting(self):
        # A split that lands entirely past a row's valid length must contribute
        # nothing rather than poisoning the merge with an empty softmax.
        S, topk, B = 4096, 640, 32
        q, kv, g = _qkv(B, S, 8, seed=99)
        idx = _random_indices(B, topk, S, g, pad_frac=0.5)
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        ref = sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, attn_sink=sink)
        for splits in (2, 10):
            with self.subTest(splits=splits):
                _assert_matches(
                    self,
                    sparse_mla_prefill(
                        q, kv, idx, SM_SCALE, D_V, attn_sink=sink, splits=splits
                    ),
                    ref,
                    f"ragged split S={splits}",
                    cos_min=0.99999,
                    max_abs=0.01,
                )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestNativePagedFP8(CustomTestCase):
    """The gather that hands the stored fp8 to the tensor core unconverted."""

    def _entry(self):
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _has_fp8_mma,
            sparse_mla_prefill_paged_fp8_native,
        )

        if not _has_fp8_mma():
            self.skipTest("device has no native fp8 mma (sm_121 upcasts)")
        return sparse_mla_prefill_paged_fp8_native

    def test_matches_bf16_gather(self):
        native = self._entry()
        S, PAGE, T, topk = 2048, 256, 512, 640
        pool, kv = _build_paged_fp8_pool(S, PAGE, seed=11)
        for h in (8, 16):
            with self.subTest(h=h):
                g = torch.Generator(device="cuda").manual_seed(h)
                q = torch.randn(
                    T, h, 512, dtype=torch.bfloat16, device="cuda", generator=g
                )
                idx = _random_indices(T, topk, S, g)
                sink = torch.randn(h, dtype=torch.float32, device="cuda", generator=g)
                # Quantising Q is what this path costs; the KV side is exact, so
                # the tolerance is looser than the bf16 gather's but bounded.
                _assert_matches(
                    self,
                    native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink),
                    sparse_mla_prefill(q, kv, idx, SM_SCALE, 512, attn_sink=sink),
                    f"native paged fp8 h={h}",
                    cos_min=0.999,
                    # 0.06 was set on an RTX 5080/5090. The same call on an RTX
                    # PRO 6000 Blackwell measures 0.0732 at h=8 and 0.0586 at
                    # h=16 -- identical with the merge at 4 warps and at 2, and
                    # with `splits=1` so the merge is not even called, i.e. it
                    # is the part, not any change here. `cos` is unmoved at
                    # 0.999465, which is what says the answer is right and only
                    # the last bits differ. Raised to cover the widest sm_120
                    # part measured, with 10% of headroom and no more, so a real
                    # drift still trips it.
                    max_abs=0.08,
                )

    def test_split_matches_unsplit(self):
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        pool, _ = _build_paged_fp8_pool(S, PAGE, seed=5)
        for B in (1, 32):
            for splits in (1, 4, 10):
                with self.subTest(B=B, splits=splits):
                    g = torch.Generator(device="cuda").manual_seed(B + splits)
                    q = torch.randn(
                        B, 8, 512, dtype=torch.bfloat16, device="cuda", generator=g
                    )
                    idx = _random_indices(B, topk, S, g)
                    sink = torch.randn(
                        8, dtype=torch.float32, device="cuda", generator=g
                    )
                    ref = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink)
                    got = native(
                        q, pool, idx, SM_SCALE, PAGE, attn_sink=sink, splits=splits
                    )
                    if splits == 1:
                        self.assertTrue(
                            torch.equal(ref, got),
                            f"native splits=1 diverged at B={B}",
                        )
                    else:
                        _assert_matches(
                            self, got, ref, f"native split B={B} S={splits}",
                            cos_min=0.99999, max_abs=0.01,
                        )

    def test_head_tile_matches_untiled(self):
        """Head tiling is a pure indexing change -- same head, same candidate
        order, different program -- so every tile must agree with the untiled
        launch bit for bit.

        Two things have to be held still for that claim to mean anything.

        **BLOCK_N.** The wrapper pins a narrower BLOCK_N when it tiles (see
        `_PINNED_NATIVE_TILED_BN`), and BLOCK_N is the softmax tile width, so
        letting it move compares two different reductions. Hence the explicit
        `config` on both arms. Without it this test passes on an unswept device,
        where no pin fires, and fails on sm_120.

        **The tile layout family.** Triton gives a `[BLOCK_H, BLOCK_N]` tile a
        `warpsPerCTA` of `[4, 1]` at BLOCK_H=64 but `[1, 4]` at BLOCK_H<=32, so
        the row reduction `tl.sum(p, axis=1)` is intra-warp at 64 and a
        cross-warp tree below it. Same arithmetic, different summation order,
        so the last bf16 bit moves across that boundary and only across it --
        8, 16 and 32 agree with each other exactly. That is why the bitwise
        assertion runs inside `_LAYOUT_FAMILY` and the 64 case is covered by
        `test_head_tile_across_layout_families` instead. (On sm_90 the same
        boundary also flips mma.sync to wgmma, whose M is fixed at 64. That is a
        second effect at the same place, not the cause: sm_120 has no wgmma and
        the boundary is still there.)
        """
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        CFG = (64, 4, 2)  # (BLOCK_N, warps, stages), pinned so only BLOCK_H moves
        pool, _ = _build_paged_fp8_pool(S, PAGE, seed=13)
        for h in (16, 32, 64):
            g = torch.Generator(device="cuda").manual_seed(h)
            q = torch.randn(
                96, h, 512, dtype=torch.bfloat16, device="cuda", generator=g
            )
            idx = _random_indices(96, topk, S, g)
            sink = torch.randn(h, dtype=torch.float32, device="cuda", generator=g)
            family = [t for t in (8, 16, 32) if t <= min(h, _LAYOUT_FAMILY)]
            ref = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                         block_h=family[-1], config=CFG)
            for tile in family[:-1]:
                with self.subTest(h=h, block_h=tile):
                    got = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                                 block_h=tile, config=CFG)
                    self.assertTrue(
                        torch.equal(ref, got),
                        f"native head tile {tile} diverged from {family[-1]} "
                        f"at h={h}, inside one layout family",
                    )

    def test_head_tile_across_layout_families(self):
        """Across the BLOCK_H=64 layout boundary the outputs are not bitwise --
        the row reduction changes order -- but tiling must still not *move* the
        error. Both tiles have to land on the same distance from the reference.
        """
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        CFG = (64, 4, 2)
        pool, kv = _build_paged_fp8_pool(S, PAGE, seed=29)
        g = torch.Generator(device="cuda").manual_seed(31)
        q = torch.randn(96, 64, 512, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = _random_indices(96, topk, S, g)
        sink = torch.randn(64, dtype=torch.float32, device="cuda", generator=g)
        ref = sparse_mla_prefill(q, kv, idx, SM_SCALE, 512, attn_sink=sink)

        wide = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                      block_h=64, config=CFG)
        narrow = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                        block_h=16, config=CFG)
        # A reduction-order difference, so a handful of bf16 ULP at most.
        _assert_matches(self, narrow, wide, "native BLOCK_H 16 vs 64",
                        cos_min=0.99999, max_abs=0.02)

        def err(o):
            c = torch.nn.functional.cosine_similarity(
                o.float().flatten(), ref.float().flatten(), dim=0).item()
            return round(c, 6), round((o.float() - ref.float()).abs().max().item(), 6)

        self.assertEqual(
            err(narrow), err(wide),
            "head tiling moved the error against the bf16 gather; it must only "
            "reorder the reduction, not change the result",
        )

    def test_head_tile_matches_untiled_split(self):
        """Same, through the split-K decode kernel. The partials are addressed
        as `mid_o[t, h, s, :]` with an absolute h, so tiling must leave both the
        layout and the merge untouched. BLOCK_N is pinned, and the tiles compared
        stay inside `_LAYOUT_FAMILY`, for the reasons given above."""
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        CFG = (64, 4, 2)
        pool, _ = _build_paged_fp8_pool(S, PAGE, seed=17)
        for B in (1, 32):
            for splits in (1, 4, 10):
                with self.subTest(B=B, splits=splits):
                    g = torch.Generator(device="cuda").manual_seed(B + splits)
                    q = torch.randn(
                        B, 64, 512, dtype=torch.bfloat16, device="cuda", generator=g
                    )
                    idx = _random_indices(B, topk, S, g)
                    sink = torch.randn(
                        64, dtype=torch.float32, device="cuda", generator=g
                    )
                    ref = native(
                        q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                        splits=splits, block_h=_LAYOUT_FAMILY, config=CFG,
                    )
                    got = native(
                        q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                        splits=splits, block_h=16, config=CFG,
                    )
                    self.assertTrue(
                        torch.equal(ref, got),
                        f"native split head tile diverged at B={B} S={splits}",
                    )

    def test_tiled_block_n_is_not_bitwise(self):
        """The BLOCK_N the wrapper pins when it tiles is a *separate* change
        from tiling, and it is deliberately not bitwise: BLOCK_N is the softmax
        tile width, so narrowing it moves the reduction boundaries. Asserted so
        that nobody later "fixes" the divergence the default path shows on a
        swept device by reverting the pin.
        """
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        cap = torch.cuda.get_device_capability()
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _PINNED_NATIVE_TILED_BN,
        )

        if cap not in _PINNED_NATIVE_TILED_BN:
            self.skipTest("no BLOCK_N pin on this device, nothing to separate")
        pool, _ = _build_paged_fp8_pool(S, PAGE, seed=19)
        g = torch.Generator(device="cuda").manual_seed(23)
        q = torch.randn(64, 64, 512, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = _random_indices(64, topk, S, g)
        sink = torch.randn(64, dtype=torch.float32, device="cuda", generator=g)
        wide = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink,
                      block_h=16, config=(64, 4, 2))
        narrow = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink, block_h=16)
        self.assertFalse(
            torch.equal(wide, narrow),
            "BLOCK_N pin did not fire -- the tiled path is not using it",
        )
        # Not bitwise, but the two must still be the same computation.
        _assert_matches(self, narrow, wide, "tiled BLOCK_N 32 vs 64",
                        cos_min=0.9999, max_abs=0.02)

    def test_head_tile_default_is_pinned_only_where_swept(self):
        """An arch with no `_PINNED_NATIVE_HEAD_TILE` entry must keep the
        untiled tile, so enabling this cannot change a device nobody measured.
        """
        import triton

        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _NATIVE_BLOCK_H,
            _PINNED_NATIVE_HEAD_TILE,
            _native_head_tile,
        )

        mono = max(_NATIVE_BLOCK_H or 0, triton.next_power_of_2(64))
        unswept = (7, 0)
        self.assertNotIn(unswept, _PINNED_NATIVE_HEAD_TILE)
        # The table is keyed by capability, so drive the resolver directly
        # rather than through a device we do not have.
        self.assertEqual(
            _PINNED_NATIVE_HEAD_TILE.get(unswept, {}).get(64, mono), mono
        )
        self.assertEqual(_PINNED_NATIVE_HEAD_TILE[(12, 0)][64], 16)
        # An explicit override is taken literally but never exceeds the tile
        # that covers every head.
        self.assertEqual(_native_head_tile(torch.cuda.current_device(), 64,
                                           mono, override=128), mono)
        self.assertEqual(_native_head_tile(torch.cuda.current_device(), 64,
                                           mono, override=16), 16)

    def test_sliced_pool_matches_contiguous(self):
        """SGLang does not hand over a contiguous pool.

        `deepseek_v4_backend` passes
        `swa_k_cache[:, : swa_window_size * k_cache_total_dim]` -- a slice along
        dim 1 -- so consecutive pages sit `stride(0)` bytes apart in a wider
        parent buffer, not `shape[-1]`. Every synthetic pool in this file is
        contiguous, so nothing here caught it until a server refused to start
        with `AssertionError: the paged cache must be contiguous`. Widen the
        pool, slice it back, and the two must agree bit for bit.
        """
        native = self._entry()
        S, PAGE, topk = 2048, 256, 640
        pool, _ = _build_paged_fp8_pool(S, PAGE, seed=37)
        pages, width = pool.shape
        # Same bytes, but each page now starts 512 B further apart.
        wide = torch.zeros(pages, width + 512, dtype=pool.dtype, device="cuda")
        wide[:, :width] = pool
        sliced = wide[:, :width]
        self.assertFalse(sliced.is_contiguous())
        self.assertEqual(sliced.stride(0), width + 512)

        g = torch.Generator(device="cuda").manual_seed(41)
        q = torch.randn(64, 64, 512, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = _random_indices(64, topk, S, g)
        sink = torch.randn(64, dtype=torch.float32, device="cuda", generator=g)
        ref = native(q, pool, idx, SM_SCALE, PAGE, attn_sink=sink)
        got = native(q, sliced, idx, SM_SCALE, PAGE, attn_sink=sink)
        self.assertTrue(
            torch.equal(ref, got),
            "a sliced (non-contiguous) pool gave a different answer",
        )

    def test_fp8_gate_excludes_sm121(self):
        # sm_121 accepts fp8 operands but software-emulates them, so the gate is
        # set membership rather than a >= comparison. _PINNED carries a (12, 1)
        # entry, which is exactly the trap this guards.
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
            _FP8_MMA_CAPS,
        )

        self.assertIn((12, 0), _FP8_MMA_CAPS)
        self.assertNotIn((12, 1), _FP8_MMA_CAPS)
