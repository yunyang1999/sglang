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

    def test_no_rope_tail_refused_by_fast_paths(self):
        # The union and dense-prefix kernels build the tail with tl.arange too,
        # so a zero-width tail must be refused up front, not compiled.
        T, topk, S, d = 256, 512, 1024, 512
        g = torch.Generator(device="cuda").manual_seed(81)
        q = torch.randn(T, 8, d, dtype=torch.bfloat16, device="cuda", generator=g)
        kv = torch.randn(S, d, dtype=torch.bfloat16, device="cuda", generator=g)
        idx = _random_indices(T, topk, S, g)
        for kwargs in ({"union": 2}, {"dense": True}):
            with self.assertRaisesRegex(ValueError, "non-empty rope tail"):
                sparse_mla_prefill(q, kv, idx, SM_SCALE, d, **kwargs)

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

    def test_attn_sink_refused_by_fast_paths(self):
        # The union and dense-prefix kernels have no sink term; taking one and
        # ignoring it would be a silent accuracy loss.
        T, topk, S = 256, 512, 1024
        q, kv, g = _qkv(T, S, 8, seed=92)
        idx = _random_indices(T, topk, S, g)
        sink = torch.randn(8, dtype=torch.float32, device="cuda", generator=g)
        for kwargs in ({"union": 2}, {"dense": True}):
            with self.assertRaisesRegex(ValueError, "base path only"):
                sparse_mla_prefill(q, kv, idx, SM_SCALE, D_V, attn_sink=sink, **kwargs)

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
