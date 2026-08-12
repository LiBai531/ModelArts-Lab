"""MXFP4 KV cache: custom spec that drives KV-cache allocation, no kv_cache_dtype hack.

Enabled via additional-config ``enable_mxfp4_kv: true``. The KV cache page is
allocated from the spec's ``real_page_size_bytes`` = packed float4 data
(head//2) + per-32-group e8m0 scale (head//32), kept as separate regions of
one page and managed by the same block table. This mirrors the GLM-5.2 SFA
``AscendMLAAttentionSpec`` approach (spec drives allocation directly), so
``kv_cache_dtype`` stays at the vLLM default and none of the previous
monkey-patches (CacheConfig swap, KVQuantMode injection, get_kv_quant_mode,
real_page_size_bytes) are needed.
"""

import math
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1 import kv_cache_interface as kv_iface

logger = init_logger(__name__)


def mxfp4_kv_cache_data_dim(head_size: int) -> int:
    """Packed float4 data bytes per head (2 fp4 values per byte)."""
    return head_size // 2


def mxfp4_kv_cache_scale_dim(head_size: int) -> int:
    """Per-32-element-group e8m0 scale count per head."""
    return head_size // 32


@dataclass(frozen=True, kw_only=True)
class AscendFullAttentionC4Spec(kv_iface.FullAttentionSpec):
    """Full attention spec for C4 (MXFP4) KV cache.

    ``head_size`` / ``head_size_v`` are the packed float4 data dims
    (head//2); ``scale_dim`` / ``scale_dim_v`` are the per-32-group e8m0
    scale dims (head//32). The page is data + scale (separate regions, one
    block table), so ``real_page_size_bytes`` returns the exact MXFP4 layout
    and vLLM allocates the correct size without touching kv_cache_dtype.

    Inherits FullAttentionSpec, so the default FullAttentionManager handles
    it (KVCacheSpecRegistry walks the MRO).
    """

    scale_dim: int = 0
    scale_dim_v: int = 0
    scale_dtype: torch.dtype = torch.uint8

    def __post_init__(self):
        super().__post_init__()
        if self.scale_dim_v == 0:
            object.__setattr__(self, "scale_dim_v", self.scale_dim)

    @property
    def real_page_size_bytes(self) -> int:
        if self.scale_dim == 0:
            # Not an MXFP4 layout: fall back to the plain full-attention page.
            return super().real_page_size_bytes
        data_bytes = (
            self.head_size + self.head_size_v
        ) * get_dtype_size(self.dtype)
        scale_bytes = (
            self.scale_dim + self.scale_dim_v
        ) * get_dtype_size(self.scale_dtype)
        return self.block_size * self.num_kv_heads * (data_bytes + scale_bytes)

    @classmethod
    def merge(cls, specs):
        """Override to preserve C4-specific fields.

        FullAttentionSpec.merge() constructs the merged spec without passing
        scale_dim / scale_dim_v / scale_dtype, so they default to 0 and the
        MXFP4 layout is silently lost. Restore them from the first spec.
        """
        merged = super().merge(specs)
        object.__setattr__(merged, "scale_dim", specs[0].scale_dim)
        object.__setattr__(merged, "scale_dim_v", specs[0].scale_dim_v)
        object.__setattr__(merged, "scale_dtype", specs[0].scale_dtype)
        return merged


import vllm.model_executor.models.config as _model_config_mod

_orig_verify = _model_config_mod.HybridAttentionMambaModelConfig.verify_and_update_config


@classmethod
def _c4_verify_and_update_config(cls, vllm_config):
    _orig_verify.__func__(cls, vllm_config)

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not additional_config.get("enable_mxfp4_kv", False):
        return

    model_config = vllm_config.model_config
    if model_config.use_mla:
        return

    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config

    kernel_block_size = 128
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    mamba_sizes = [
        math.prod(s) * get_dtype_size(d)
        for s, d in zip(mamba_shapes, mamba_dtypes)
    ]
    ssm_block_page_size = max(mamba_sizes)
    conv_block_page_size = min(mamba_sizes) if len(mamba_sizes) > 1 else 0

    # C4 k_data + v_data per token (uint8): both alias with ssm.
    # (k_data + v_data)_page must equal ssm_page for aliasing.
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_data_per_token = mxfp4_kv_cache_data_dim(head_size) * num_kv_heads * 2

    attn_block_size = kernel_block_size * cdiv(
        ssm_block_page_size, kernel_block_size * kv_data_per_token
    )
    assert kv_data_per_token * attn_block_size == ssm_block_page_size, (
        f"Cannot align C4 kv_data page and ssm page: "
        f"kv_data_per_token={kv_data_per_token}, "
        f"block_size={attn_block_size}, "
        f"ssm_page={ssm_block_page_size}"
    )

    cache_config.block_size = attn_block_size

    # Recompute mamba_page_size_padded using C4 page dims.
    # C4 token page = (data+scale) * 2 (k+v) * num_kv_heads, all uint8.
    # k_data + v_data are aliased with ssm; only k_scale + v_scale are
    # extra (not used by mamba).
    scale_dim = mxfp4_kv_cache_scale_dim(head_size)
    c4_token_page_size = (
        (mxfp4_kv_cache_data_dim(head_size) + scale_dim) * 2 * num_kv_heads
    )
    attn_page_size = cache_config.block_size * c4_token_page_size
    cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size

    if cache_config.enable_prefix_caching and cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size

    logger.info(
        "[mxfp4_kv] C4 aliasing: block_size=%d, kv_data_per_token=%d, "
        "ssm_page=%d, mamba_page_size_padded=%d",
        attn_block_size,
        kv_data_per_token,
        ssm_block_page_size,
        cache_config.mamba_page_size_padded,
    )


_model_config_mod.HybridAttentionMambaModelConfig.verify_and_update_config = (
    _c4_verify_and_update_config
)
