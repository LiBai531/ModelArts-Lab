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

from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm.v1 import kv_cache_interface as kv_iface


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
