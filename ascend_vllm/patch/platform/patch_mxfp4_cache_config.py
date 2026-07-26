import torch

from vllm.config import cache as cache_mod
from vllm.utils import torch_utils as tu
from vllm.v1 import kv_cache_interface as kv_iface

_orig_init = cache_mod.CacheConfig.__init__


def _new_init(self, *args, **kwargs):
    _restore = False
    if kwargs.get("cache_dtype") == "mxfp4":
        kwargs["cache_dtype"] = "nvfp4"
        _restore = True
    elif len(args) > 3 and args[3] == "mxfp4":
        # cache_dtype is the 4th init field of CacheConfig:
        # (block_size, prefix_match_unit, gpu_memory_utilization, cache_dtype)
        args = list(args)
        args[3] = "nvfp4"
        _restore = True
    _orig_init(self, *args, **kwargs)
    if _restore:
        object.__setattr__(self, "cache_dtype", "mxfp4")


cache_mod.CacheConfig.__init__ = _new_init

tu.STR_DTYPE_TO_TORCH_DTYPE["mxfp4"] = torch.uint8

_orig_is_quant = tu.is_quantized_kv_cache


def _new_is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return _orig_is_quant(kv_cache_dtype) or kv_cache_dtype == "mxfp4"

tu.is_quantized_kv_cache = _new_is_quantized_kv_cache

_KVQM = kv_iface.KVQuantMode

if not hasattr(_KVQM, "MXFP4"):
    # Use value 6: 5 is already taken by NVFP4. MXFP4 must not alias it,
    # otherwise IntEnum equality makes is_mxfp4 / is_nvfp4 collide.
    _mxfp4_member = int.__new__(_KVQM, 6)
    _mxfp4_member._name_ = "MXFP4"
    _mxfp4_member._value_ = 6
    _KVQM._member_map_["MXFP4"] = _mxfp4_member
    _KVQM._value2member_map_[6] = _mxfp4_member

_KVQM.is_mxfp4 = property(lambda self: self == _KVQM.MXFP4)


def mxfp4_kv_cache_data_dim(head_size: int) -> int:
    return head_size // 2


def mxfp4_kv_cache_scale_dim(head_size: int) -> int:
    return head_size // 32


_orig_get_kv_quant_mode = kv_iface.get_kv_quant_mode


def _new_get_kv_quant_mode(kv_cache_dtype: str):
    if kv_cache_dtype == "mxfp4":
        return _KVQM.MXFP4
    return _orig_get_kv_quant_mode(kv_cache_dtype)

kv_iface.get_kv_quant_mode = _new_get_kv_quant_mode

for _mod_name in (
    "vllm.model_executor.layers.attention.attention",
    "vllm.model_executor.layers.attention"
):
    try:
        import importlib
        _m = importlib.import_module(_mod_name)
        if hasattr(_m, "get_kv_quant_mode"):
            _m.get_kv_quant_mode = _new_get_kv_quant_mode
    except ImportError:
        pass

_get_dtype_size = tu.get_dtype_size

_AS = kv_iface.AttentionSpec
_orig_as_rpsb = _AS.real_page_size_bytes.fget


def _mxfp4_as_rpsb(self):
    if self.kv_quant_mode == _KVQM.MXFP4:
        data_dim = mxfp4_kv_cache_data_dim(self.head_size)
        scale_dim = mxfp4_kv_cache_scale_dim(self.head_size)
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * (data_dim + scale_dim)
            * _get_dtype_size(self.dtype)
        )
    return _orig_as_rpsb(self)

_AS.real_page_size_bytes = property(_mxfp4_as_rpsb)

_FAS = kv_iface.FullAttentionSpec
_orig_fas_rpsb = _FAS.real_page_size_bytes.fget


def _mxfp4_fas_rpsb(self):
    if self.kv_quant_mode == _KVQM.MXFP4:
        last_dim = (
            mxfp4_kv_cache_data_dim(self.head_size)
            + mxfp4_kv_cache_data_dim(self.head_size_v)
            + mxfp4_kv_cache_scale_dim(self.head_size)
            + mxfp4_kv_cache_scale_dim(self.head_size_v)
        )

        return (
            self.block_size
            * self.num_kv_heads
            * last_dim
            * _get_dtype_size(self.dtype)
        )
    return _orig_fas_rpsb(self)

_FAS.real_page_size_bytes = property(_mxfp4_fas_rpsb)

_SWS = kv_iface.SlidingWindowSpec
_orig_sws_rpsb = _SWS.real_page_size_bytes.fget


def _mxfp4_sws_rpsb(self):
    if self.kv_quant_mode == _KVQM.MXFP4:
        last_dim = (
            mxfp4_kv_cache_data_dim(self.head_size)
            + mxfp4_kv_cache_data_dim(self.head_size_v)
            + mxfp4_kv_cache_scale_dim(self.head_size)
            + mxfp4_kv_cache_scale_dim(self.head_size_v)
        )

        return (
            self.block_size
            * self.num_kv_heads
            * last_dim
            * _get_dtype_size(self.dtype)
        )
    return _orig_sws_rpsb(self)

_SWS.real_page_size_bytes = property(_mxfp4_sws_rpsb)