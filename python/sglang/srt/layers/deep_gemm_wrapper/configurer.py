import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_sm100_supported,
    is_sm120_supported,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # SM120 uses mma.sync block-scale kernels (no TMEM/tcgen05), added in
    # DeepGEMM#324; installed builds may predate it, so probe the entry point
    # (same idiom as the deep_gemm import probe below).
    if sm_version == 120:
        try:
            from deep_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous  # noqa: F401
        except (ImportError, AttributeError):
            return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

# DeepGEMM PR #324 supports the Blackwell-family MXFP4 path on both SM100
# and SM120. The path expects packed UE8M0 scale tensors for FP8/FP4 GEMMs.
# (The EPv2/DeepEP fp8-dispatch gates read DEEPGEMM_BLACKWELL, so SM120 is
# included here on this branch; the SGL_USE_DEEPGEMM_BMM weight path is the
# only consumer that flips with it and stays off by default.)
DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and (
    is_sm100_supported() or is_sm120_supported()
)
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
