"""MXFP4 dual-pool copy-on-write (CoW) guard.

When mxfp4 dual block pool is active, the mamba layers hold their own block-id
space (independent ``BlockPool`` sized ``ratio * num_blocks``) while the C4
attention layers use the main pool. The upstream ``copy_kv_cache_blocks_inplace``
applies every CoW copy to every distinct KV cache storage, so an attention
partial-hit CoW (block ids < ``num_blocks``) would be copied into the mamba
storage at the same offsets, corrupting the mamba state that uses independent
block ids.

The patch skips ``list`` entries (mamba state tensors) from the CoW target set
when the model mixes mamba (``list``) and attention (``tuple``) cache entries:

* **dual pool**: mamba storage is skipped -> attention CoW never touches it.
* **single pool (legacy / non-mxfp4 hybrid)**: mamba and attention alias the
  *same* backing storage, so the mamba list entry is deduplicated away by the
  remaining non-list entry anyway (``seen_storage``); behaviour is unchanged.
* **pure mamba model**: every entry is a list -> no skip, CoW applied as usual.

So the patch is safe under both pool layouts and for non-mxfp4 models. A mamba
*partial-hit* CoW (which only occurs when ``mamba_block_size > hash_block_size``,
not the default) is still not applied to the mamba pool; enabling finer-grained
mamba prefix caching will need per-pool CoW splitting.
"""

import functools
import sys

from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.worker.patch_mxfp4_cow")

_PATCH_APPLIED = False


def _patch_copy_kv_cache_blocks_inplace() -> None:
    import vllm.v1.worker.utils as vllm_worker_utils

    orig = vllm_worker_utils.copy_kv_cache_blocks_inplace
    if getattr(orig, "_modelarts_mxfp4_cow_patched", False):
        return

    @functools.wraps(orig)
    def patched(kv_caches, num_blocks, kv_cache_block_copies):
        if not kv_cache_block_copies:
            return
        entries = list(kv_caches)
        has_list = any(isinstance(e, list) for e in entries)
        has_non_list = any(not isinstance(e, list) for e in entries)
        # Only mixed models: skip the mamba (list) entries. Pure mamba keeps
        # them so its CoW still runs.
        if has_list and has_non_list:
            filtered = [e for e in entries if not isinstance(e, list)]
            skipped = len(entries) - len(filtered)
            if skipped:
                logger.debug(
                    "[mxfp4_kv] CoW: skipping %d mamba cache entries (independent "
                    "block pool); applying %d copies to %d attention entries",
                    skipped,
                    len(kv_cache_block_copies),
                    len(filtered),
                )
            if not filtered:
                return
            kv_caches = filtered
        return orig(kv_caches, num_blocks, kv_cache_block_copies)

    patched._modelarts_mxfp4_cow_patched = True
    vllm_worker_utils.copy_kv_cache_blocks_inplace = patched

    # `from vllm.v1.worker.utils import ... copy_kv_cache_blocks_inplace`
    # binds the old function object at import time; refresh those bindings if
    # the importing modules are already loaded.
    for mod_name in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "copy_kv_cache_blocks_inplace", None) is orig:
            mod.copy_kv_cache_blocks_inplace = patched


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_copy_kv_cache_blocks_inplace()
    _PATCH_APPLIED = True


apply_patch()
