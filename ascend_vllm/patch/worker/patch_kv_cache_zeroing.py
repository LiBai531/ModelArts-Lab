"""PD-separation / KV-cache zeroing alignment for the mxfp4 dual pool.

Two fixes needed for prefill-decoding disaggregation (KV transfer) with the
mxfp4 4-tuple (K, V, K_scale, V_scale) cache:

1. ``AscendKVBlockZeroer.init_meta`` asserts ``len(kv_tuple) == 2``; a C4 layer
   carries 4 tensors and the assert fails at worker start. Zeroing only needs
   the K/V *data* (stale scale * zeroed data == 0), so the patch temporarily
   presents ``(K, V)`` to the stock implementation and restores the 4-tuple
   afterwards. Both K and V data share the same block page, so the stock
   per-tensor page computation stays uniform.

2. ``SingleTypeKVCacheManager.__init__`` gates ``_record_new_block_ids`` on an
   exact ``type(spec) in (FullAttentionSpec, ...)`` list, which misses the
   ``AscendFullAttentionC4Spec`` subclass. Without it, C4 blocks are never
   recorded and the worker never zeroes them (upstream bf16 attention does).
   The patch records when ``needs_kv_cache_zeroing`` is enabled.

These patches are safe when mxfp4 is disabled: the 4-tuple branch never matches
(a 2-tuple / tensor cache is untouched), and the manager gate only flips for the
C4 spec type, which does not exist in non-mxfp4 runs.
"""

import functools

from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.worker.patch_kv_cache_zeroing")

_PATCH_APPLIED = False


def _patch_ascend_kv_block_zeroer() -> None:
    from vllm_ascend.worker.utils import AscendKVBlockZeroer

    orig = AscendKVBlockZeroer.init_meta
    if getattr(orig, "_modelarts_mxfp4_zeroing_patched", False):
        return

    @functools.wraps(orig)
    def patched_init_meta(self, *args, **kwargs):
        static_forward_context = kwargs.get("static_forward_context")
        if static_forward_context is None and len(args) >= 5:
            static_forward_context = args[4]
        swapped = []
        try:
            if static_forward_context is not None:
                for layer in static_forward_context.values():
                    kv = getattr(layer, "kv_cache", None)
                    # mxfp4 C4 cache is (K, V, K_scale, V_scale). Only K/V data
                    # need zeroing; present (K, V) so the stock len==2 assert and
                    # uniform page-size computation pass.
                    if isinstance(kv, (list, tuple)) and len(kv) == 4:
                        layer.kv_cache = kv[:2]
                        swapped.append((layer, kv))
                if swapped:
                    logger.info(
                        "[mxfp4_kv] zeroing meta: presenting K/V data of %d "
                        "C4 layer(s) to AscendKVBlockZeroer (4-tuple -> 2-tuple)",
                        len(swapped),
                    )
            return orig(self, *args, **kwargs)
        finally:
            for layer, kv in swapped:
                layer.kv_cache = kv

    patched_init_meta._modelarts_mxfp4_zeroing_patched = True
    AscendKVBlockZeroer.init_meta = patched_init_meta


def _patch_record_new_block_ids() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import (
        SingleTypeKVCacheManager,
    )

    from ascend_vllm.patch.platform.patch_mxfp4_cache_config import (
        AscendFullAttentionC4Spec,
    )

    orig = SingleTypeKVCacheManager.__init__

    @functools.wraps(orig)
    def patched_init(self, kv_cache_spec, *args, **kwargs):
        orig(self, kv_cache_spec, *args, **kwargs)
        if isinstance(kv_cache_spec, AscendFullAttentionC4Spec):
            # The exact-type gate in the base __init__ misses our subclass;
            # record new block ids whenever zeroing is enabled, mirroring the
            # upstream bf16 attention behaviour.
            self._record_new_block_ids = bool(
                kwargs.get("needs_kv_cache_zeroing", False)
            )

    SingleTypeKVCacheManager.__init__ = patched_init


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_ascend_kv_block_zeroer()
    _patch_record_new_block_ids()
    _PATCH_APPLIED = True


apply_patch()
