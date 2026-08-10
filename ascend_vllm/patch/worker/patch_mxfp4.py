import math

import torch
import torch_npu
from vllm.logger import logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode, MambaSpec

try:
    from scipy.linalg import hadamard as _hadamard
except ImportError:  # pragma: no cover
    _hadamard = None

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

from ascend_vllm.patch.platform.patch_mxfp4_cache_config import (
    AscendFullAttentionC4Spec,
    mxfp4_kv_cache_data_dim,
    mxfp4_kv_cache_scale_dim,
)

FLOAT4_E2M1FN_X2_DTYPE = getattr(
    torch_npu, "float4_e2m1fn_x2", getattr(torch, "float4_e2m1fn_x2", None)
)
MXFP4_SCALE_GROUP_SIZE = 32
SWA_INT_MAX = 2147483647


def _mxfp4_fia_v2_dequant_kwargs():
    """Constant mxfp4 dequant kwargs shared by all FIA v2 calls."""
    return {
        "key_quant_mode": 6,
        "value_quant_mode": 6,
        "key_dtype": torch.float4_e2m1fn_x2,
        "value_dtype": torch.float4_e2m1fn_x2,
        "dequant_scale_key_dtype": torch.float8_e8m0fnu,
        "dequant_scale_value_dtype": torch.float8_e8m0fnu,
    }


def _is_mxfp4_kv_enabled() -> bool:
    try:
        from vllm_ascend.ascend_config import get_ascend_config
        ascend_config = get_ascend_config()
        vllm_config = getattr(ascend_config, "vllm_config", None)
        if vllm_config is None:
            return False
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        return bool(additional_config.get("enable_mxfp4_kv", False))
    except RuntimeError:
        # ascend_config not initialized yet (patch loaded before worker init)
        return False


_ROTATION_MATRICES: dict[tuple, torch.Tensor] = {}


def _get_orthogonal_block(
    device: str = "npu", dtype: torch.dtype = torch.float32,
) -> torch.Tensor: 
    key = ("block", device, dtype)
    if key not in _ROTATION_MATRICES:
        if _hadamard is None:
            raise ImportError(
                "mxfp4 KV cache requires scipy for the Hadamard rotation "
                "matrix (pip install scipy)"
            )
        _ROTATION_MATRICES[key] = (
            torch.tensor(_hadamard(MXFP4_SCALE_GROUP_SIZE, dtype=float), dtype=dtype, device=device)
            / math.sqrt(MXFP4_SCALE_GROUP_SIZE)
        )

    return _ROTATION_MATRICES[key]


AscendAttentionBackendImpl._hadamard_32 = None

_orig_attn_init = AscendAttentionBackendImpl.__init__


def _mxfp4_attn_init(
    self, 
    num_heads, head_size, scale, num_kv_heads, alibi_slopes,
    sliding_window, kv_cache_dtype, logits_soft_cap, attn_type,
    kv_sharing_target_layer_name, sinks=None, **kwargs,
):
    _orig_attn_init(
        self, num_heads, head_size, scale, num_kv_heads, alibi_slopes,
        sliding_window, kv_cache_dtype, logits_soft_cap, attn_type,
        kv_sharing_target_layer_name, sinks=sinks, **kwargs,
    )
    self.enable_mxfp4_kv_cache = _is_mxfp4_kv_enabled()
    self.mxfp4_k_scale_cache = None
    self.mxfp4_v_scale_cache = None
    # Per-layer, per-graph-size rotation buffers. Keyed by num_tokens because
    # one layer is captured at multiple graph sizes; stored on the instance
    # (not a global pool) so layers never share a buffer.
    self.mxfp4_query_rot_buffers: dict[int, torch.Tensor] = {}

AscendAttentionBackendImpl.__init__ = _mxfp4_attn_init

_orig_pwal = AscendAttentionBackendImpl.process_weights_after_loading


def _mxfp4_pwal(self, act_dtype: torch.dtype):
    _orig_pwal(self, act_dtype)
    if getattr(self, "enable_mxfp4_kv_cache", False):
        if AscendAttentionBackendImpl._hadamard_32 is None:
            AscendAttentionBackendImpl._hadamard_32 = _get_orthogonal_block(
                device="npu", dtype=torch.bfloat16
            )

AscendAttentionBackendImpl.process_weights_after_loading = _mxfp4_pwal


def _rotate(self, x: torch.Tensor) -> torch.Tensor:
    Q = AscendAttentionBackendImpl._hadamard_32
    if Q is None or Q.dtype != x.dtype or Q.device != x.device:
        Q = _get_orthogonal_block(device=x.device, dtype=x.dtype)
    original_shape = x.shape
    x = x.reshape(-1, MXFP4_SCALE_GROUP_SIZE)
    x = x @ Q
    return x.reshape(original_shape)

AscendAttentionBackendImpl._rotate = _rotate


def _quantize_kv_to_mxfp4(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    num_actual_tokens: int,
) -> tuple:
    actual_key = key[:num_actual_tokens]
    actual_value = value[:num_actual_tokens]

    # K/V write path is eager (outside the ACL graph capture region: it uses
    # per-step slot_mapping), so a device check is safe here. Use the
    # per-(device,dtype) cached Hadamard block (built eagerly in pwal / first
    # forward) so multi-device (TP) is correct and no tensor is created during
    # graph capture.
    Q = AscendAttentionBackendImpl._hadamard_32
    if Q is None or Q.dtype != actual_key.dtype or Q.device != actual_key.device:
        Q = _get_orthogonal_block(
            device=actual_key.device, dtype=actual_key.dtype
        )

    key_mxfp4, k_scales = torch_npu.npu_rotate_quant(
        actual_key,
        Q,
        dst_dtype=FLOAT4_E2M1FN_X2_DTYPE,
        axis=-1,
        round_mode="round",
        scale_alg=0,
        dst_type_max=0.0,
        transpose_y=False
    )

    value_mxfp4, v_scales = torch_npu.npu_dynamic_mx_quant(
        actual_value, dst_type=FLOAT4_E2M1FN_X2_DTYPE, round_mode="round"
    )

    k_scales = k_scales.view(k_scales.shape[0], k_scales.shape[1], -1)
    v_scales = v_scales.view(v_scales.shape[0], v_scales.shape[1], -1)
    return key_mxfp4, value_mxfp4, k_scales, v_scales

AscendAttentionBackendImpl._quantize_kv_to_mxfp4 = _quantize_kv_to_mxfp4


def _scatter_mxfp4_kv_and_scales(
    self,
    key_mxfp4, value_mxfp4,
    k_scales, v_scales,
    kv_cache, attn_metadata
):
    if not isinstance(kv_cache, list | tuple) or len(kv_cache) < 2:
        return
    
    if kv_cache[0] is not self.key_cache:
        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
    if len(kv_cache) >= 4 and kv_cache[2] is not self.mxfp4_k_scale_cache:
        self.mxfp4_k_scale_cache, self.mxfp4_v_scale_cache = kv_cache[2], kv_cache[3]

    slots = attn_metadata.slot_mapping
    num_actual = attn_metadata.num_actual_tokens

    torch_npu.npu_scatter_pa_kv_cache(
        key=key_mxfp4[:num_actual].view(torch.uint8).contiguous(),
        value=value_mxfp4[:num_actual].view(torch.uint8).contiguous(),
        key_cache=self.key_cache,
        value_cache=self.value_cache,
        slot_mapping=slots[:num_actual].contiguous(),
        cache_mode="Norm"
    )

    if self.mxfp4_k_scale_cache is not None:
        torch_npu.npu_scatter_pa_kv_cache(
            key=k_scales[:num_actual].view(torch.uint8).contiguous(),
            value=v_scales[:num_actual].view(torch.uint8).contiguous(),
            key_cache=self.mxfp4_k_scale_cache,
            value_cache=self.mxfp4_v_scale_cache,
            slot_mapping=slots[:num_actual].contiguous(),
            cache_mode="Norm"
        )

AscendAttentionBackendImpl._scatter_mxfp4_kv_and_scales = _scatter_mxfp4_kv_and_scales

_orig_attn_forward = AscendAttentionBackendImpl.forward


def _mxfp4_forward(
    self, layer, query, key, value, kv_cache, attn_metadata,
    output=None, output_scale=None, output_block_scale=None
):
    if getattr(self, "enable_mxfp4_kv_cache", False) and attn_metadata is not None:
        # Record the layer name so graph replay can match metadata by name
        # instead of relying on attn_metadata iteration order (mirrors
        # upstream layer-aware replay for mixed-attention models).
        self._layer_name = layer.layer_name
        return self._forward_mxfp4(
            layer, query, key, value, kv_cache, attn_metadata, output
        )
    return _orig_attn_forward(
        self, layer, query, key, value, kv_cache, attn_metadata,
        output=output, output_scale=output_scale,
        output_block_scale=output_block_scale
    )

AscendAttentionBackendImpl.forward = _mxfp4_forward


def _forward_mxfp4(
    self, layer, query, key, value, kv_cache, attn_metadata, output,
) -> torch.Tensor:
    if self.key_cache is None and kv_cache is not None:
        if (
            isinstance(kv_cache, torch.Tensor)
            and kv_cache.dim() > 0
            and kv_cache.shape[0] == 2
            or isinstance(kv_cache, list | tuple)
            and len(kv_cache) >= 2
        ):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if len(kv_cache) >= 4:
                self.mxfp4_k_scale_cache = kv_cache[2]
                self.mxfp4_v_scale_cache = kv_cache[3]
    if (
        self.mxfp4_k_scale_cache is None
        and self.key_cache is not None
        and self.key_cache.dim() == 4
    ):
        num_blocks, block_size = self.key_cache.shape[0], self.key_cache.shape[1]
        scale_dim = self.key_cache.shape[-1] // 16
        self.mxfp4_k_scale_cache = torch.zeros(
            num_blocks, block_size, self.num_kv_heads, scale_dim,
            dtype=torch.uint8, device=self.key_cache.device,
        )
        self.mxfp4_v_scale_cache = torch.zeros(
            num_blocks, block_size, self.num_kv_heads, scale_dim,
            dtype=torch.uint8, device=self.key_cache.device,
        )

    float_key, float_value = None, None
    if key is not None and value is not None:
        if attn_metadata.attn_state not in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        ):
            float_key, float_value = key, value
        key, value, k_scales, v_scales = self._quantize_kv_to_mxfp4(
            key, value, attn_metadata.num_actual_tokens
        )
        self._scatter_mxfp4_kv_and_scales(
            key, value, k_scales, v_scales, kv_cache, attn_metadata
        )
    
    if attn_metadata.attn_state in (
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding
    ):
        if _EXTRA_CTX.capturing:
            attn_output, num_tokens = self.full_graph_mxfp4_decode(
                query, attn_metadata, output
            )
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        return self._forward_mxfp4_decode(query, attn_metadata, output)
    elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
        return self._forward_mxfp4_chunked_prefill(
            query, float_key, float_value, attn_metadata, output
        )
    else:
        return self._forward_mxfp4_fused_infer_attention(
            query, float_key, float_value, attn_metadata, output
        )

AscendAttentionBackendImpl._forward_mxfp4 = _forward_mxfp4


def full_graph_mxfp4_decode(
    self, query, attn_metadata, output
) -> tuple:
    num_block, block_size, _, _ = self.key_cache.shape
    key_3d = self.key_cache.view(num_block, block_size, -1)
    value_3d = self.value_cache.view(num_block, block_size, -1)

    if attn_metadata.attn_state in (
        AscendAttentionState.SpecDecoding,
        AscendAttentionState.ChunkedPrefill
    ):
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q[:num_decodes]
        num_tokens = int(actual_seq_qlen[-1])
        batch_size = num_decodes
        block_tables = attn_metadata.block_tables[:num_decodes]
        seq_lens_kv = attn_metadata.seq_lens_list[:num_decodes]

    else:
        batch_size = len(attn_metadata.seq_lens_list)
        num_tokens = batch_size
        actual_seq_qlen = torch.arange(1, batch_size + 1, dtype=torch.int32)
        block_tables = attn_metadata.block_tables[:batch_size]
        seq_lens_kv = attn_metadata.seq_lens_list[:batch_size]

    if _EXTRA_CTX.is_draft_model:
        # Align with upstream full_graph_fia: derive from attn_metadata.causal
        # instead of forcing False, so a q_len>1 draft stays correct. For
        # sequential MTP (q_len=1) this is equivalent to mode 0.
        use_causal_mask = attn_metadata.causal
    else:
        use_causal_mask = num_tokens > batch_size

    graph_sparse_mode = 3 if use_causal_mask else 0
    graph_attn_mask = attn_metadata.attn_mask if use_causal_mask else None

    if _EXTRA_CTX.is_draft_model:
        graph_params = get_draft_graph_params()
    else:
        graph_params = get_graph_params()

    workspace = graph_params.workspaces.get(num_tokens)
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)

    # One rotation buffer per (num_tokens) per layer instance: a layer is
    # captured at multiple graph sizes, and different layers must never share
    # a buffer (they would overwrite each other on replay).
    _query_rot = self.mxfp4_query_rot_buffers.get(num_tokens)
    if _query_rot is None:
        _query_rot = torch.zeros_like(query[:num_tokens])
        self.mxfp4_query_rot_buffers[num_tokens] = _query_rot

    k_scale = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
    v_scale = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)

    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query=query[:num_tokens],
            key=key_3d,
            value=value_3d,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            block_table=block_tables,
            atten_mask=graph_attn_mask,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=seq_lens_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            softmax_scale=self.scale,
            **_mxfp4_fia_v2_dequant_kwargs(),
            sparse_mode=graph_sparse_mode
        )
        if _EXTRA_CTX.is_draft_model:
            update_draft_graph_params_workspaces(num_tokens, workspace)
        else:
            update_graph_params_workspaces(num_tokens, workspace)

    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
    graph_params.events[num_tokens].append(event)
    graph_params.attn_params[num_tokens].append(
        (
            weak_ref_tensors(_query_rot),
            weak_ref_tensors(key_3d),
            weak_ref_tensors(value_3d),
            weak_ref_tensors(block_tables),
            weak_ref_tensors(graph_attn_mask) if graph_attn_mask is not None else None,
            block_size,
            seq_lens_kv,
            actual_seq_qlen,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            graph_sparse_mode,
            SWA_INT_MAX,
            0,
            self.mxfp4_k_scale_cache,
            None,
            self.mxfp4_v_scale_cache,
            None,
            weak_ref_tensors(_query_rot),
            # Weak ref to the live query tensor: update_graph_params later
            # re-dereferences it to rotate the current step's query into the
            # fixed-address _query_rot buffer. This relies on vLLM's graph
            # capture semantics where intermediate tensors (query) are stable
            # buffers whose contents are overwritten each step, NOT recreated
            # per forward. If upstream ever switches to per-step allocation,
            # this weak ref may resolve to None and rotation would be skipped.
            weak_ref_tensors(query),
            self._graph_metadata_layer_name(),
        )
    )

    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score_v2.out(
        query=_query_rot,
        key=key_3d,
        value=value_3d,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        block_table=block_tables,
        atten_mask=graph_attn_mask,
        input_layout="TND",
        block_size=block_size,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=seq_lens_kv,
        num_query_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        softmax_scale=self.scale,
        **_mxfp4_fia_v2_dequant_kwargs(),
        sparse_mode=graph_sparse_mode,
        workspace=workspace,
        out=[output, softmax_lse]
    )

    output = output.view(num_tokens, self.num_heads, self.head_size)
    handle = torch.npu.graph_task_group_end(stream)
    graph_params.handles[num_tokens].append(handle)
    return output, num_tokens

AscendAttentionBackendImpl.full_graph_mxfp4_decode = full_graph_mxfp4_decode


def _forward_mxfp4_decode(self, query, attn_metadata, output) -> torch.Tensor:
    num_block, block_size, _, _ = self.key_cache.shape
    key_3d = self.key_cache.view(num_block, block_size, -1)
    value_3d = self.value_cache.view(num_block, block_size, -1)

    if attn_metadata.attn_state == AscendAttentionState.SpecDecoding:
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q[:num_decodes]
        num_valid_tokens = int(actual_seq_qlen[-1])
        block_tables = attn_metadata.block_tables[:num_decodes]
        seq_lens_kv = attn_metadata.seq_lens_list[:num_decodes]

        k_scale = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
        v_scale = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)

        query_rot = self._rotate(query[:num_valid_tokens])
        use_causal_mask = num_valid_tokens > num_decodes

        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=query_rot,
            key=key_3d,
            value=value_3d,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            block_table=block_tables,
            atten_mask=attn_metadata.attn_mask if use_causal_mask else None,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=seq_lens_kv,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            **_mxfp4_fia_v2_dequant_kwargs(),
            sparse_mode=3 if use_causal_mask else 0
        )
        attn_output = attn_output.view(num_valid_tokens, self.num_heads, self.head_size)
        output[:num_valid_tokens] = attn_output

    else:
        batch_size = len(attn_metadata.seq_lens_list)
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        block_tables = attn_metadata.block_tables[:batch_size]
        seq_lens_kv = attn_metadata.seq_lens_list[:batch_size]

        k_scale = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
        v_scale = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)

        query_rot = self._rotate(query[:batch_size])

        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=query_rot,
            key=key_3d,
            value=value_3d,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            block_table=block_tables,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=seq_lens_kv,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            **_mxfp4_fia_v2_dequant_kwargs(),
            sparse_mode=0
        )
        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output

    return output

AscendAttentionBackendImpl._forward_mxfp4_decode = _forward_mxfp4_decode


def _forward_mxfp4_chunked_prefill(
    self, query, float_key, float_value, attn_metadata, output, 
) -> torch.Tensor:
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_decodes = attn_metadata.num_decodes
    actual_seq_qlen = attn_metadata.actual_seq_lengths_q
    num_tokens = int(actual_seq_qlen[-1])
    num_valid_decode_tokens = 0

    if num_decode_tokens > 0:
        if _EXTRA_CTX.capturing:
            attn_output, num_valid_decode_tokens = self.full_graph_mxfp4_decode(
                query, attn_metadata, output
            )
            output[:num_valid_decode_tokens] = attn_output[:num_valid_decode_tokens]
        else:
            num_block, block_size, _, _ = self.key_cache.shape
            key_3d = self.key_cache.view(num_block, block_size, -1)
            value_3d = self.value_cache.view(num_block, block_size, -1)

            block_tables_decode = attn_metadata.block_tables[:num_decodes]
            seq_lens_decode = attn_metadata.seq_lens_list[:num_decodes]
            batch_size_decode = num_decodes

            k_scale = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
            v_scale = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)

            actual_seq_qlen_decode = actual_seq_qlen[:batch_size_decode]
            num_valid_decode_tokens = int(actual_seq_qlen_decode[-1])
            query_rot = self._rotate(query[:num_valid_decode_tokens])
            use_causal_mask = num_valid_decode_tokens > batch_size_decode

            attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=query_rot,
                key=key_3d,
                value=value_3d,
                dequant_scale_key=k_scale,
                dequant_scale_value=v_scale,
                block_table=block_tables_decode,
                atten_mask=attn_metadata.attn_mask if use_causal_mask else None,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen_decode,
                actual_seq_kvlen=seq_lens_decode,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                **_mxfp4_fia_v2_dequant_kwargs(),
                sparse_mode=3 if use_causal_mask else 0
            )

            attn_out = attn_out.view(num_valid_decode_tokens, self.num_heads, self.head_size)
            output[:num_valid_decode_tokens] = attn_out
    
    if attn_metadata.num_prefills > 0:
        prefill_q = query[num_valid_decode_tokens:num_tokens]
        prefill_seq_qlen = [
            actual_seq_qlen[i] - num_valid_decode_tokens
            for i in range(num_decodes, len(actual_seq_qlen))
        ]

        all_new_prefill = True
        for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
            q_start = actual_seq_qlen[i - 1] if i > 0 else 0
            qlen_i = actual_seq_qlen[i] - q_start
            if attn_metadata.seq_lens_list[i] > qlen_i:
                all_new_prefill = False
                break

        if all_new_prefill and float_key is not None and float_value is not None:
            prefill_k = float_key[num_valid_decode_tokens:num_tokens]
            prefill_v = float_value[num_valid_decode_tokens:num_tokens]
            prefill_seq_kvlen = prefill_seq_qlen
            cache_block_size = self.key_cache.shape[1]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3
            )
        else:
            num_block, block_size, _, _ = self.key_cache.shape
            key_bnsd = self.key_cache.view(num_block, block_size, -1)
            value_bnsd = self.value_cache.view(num_block, block_size, -1)

            prefill_block_tables = attn_metadata.block_tables[num_decodes:]
            prefill_seq_lens_kv = attn_metadata.seq_lens_list[num_decodes:]

            k_scale_pf = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
            v_scale_pf = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)
            prefill_actual_seq_qlen = torch.tensor(
                prefill_seq_qlen, dtype=torch.int32
            )
            prefill_q_rot = self._rotate(prefill_q)
            attn_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query=prefill_q_rot,
                key=key_bnsd,
                value=value_bnsd,
                dequant_scale_key=k_scale_pf,
                dequant_scale_value=v_scale_pf,
                block_table=prefill_block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=prefill_actual_seq_qlen,
                actual_seq_kvlen=prefill_seq_lens_kv,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                **_mxfp4_fia_v2_dequant_kwargs(),
                sparse_mode=3
            )
        
        n_prefill = num_tokens - num_valid_decode_tokens
        attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
        output[num_valid_decode_tokens:num_tokens] = attn_out[:n_prefill]

    return output

AscendAttentionBackendImpl._forward_mxfp4_chunked_prefill = _forward_mxfp4_chunked_prefill


def _forward_mxfp4_fused_infer_attention(
    self, query, float_key, float_value, attn_metadata, output, 
) -> torch.Tensor:
    actual_seq_qlen = attn_metadata.actual_seq_lengths_q
    num_tokens = int(actual_seq_qlen[-1])
    query = query[:num_tokens]

    from vllm.v1.attention.backend import AttentionType

    if (
        attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
        and self.attn_type != AttentionType.ENCODER_DECODER
    ):
        # No cached prefix: attend over the freshly computed float K/V.
        key = float_key[:num_tokens]
        value = float_value[:num_tokens]
        actual_seq_lengths_kv = actual_seq_qlen
        block_size = self.key_cache.shape[1]
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
    else:
        # PrefillCacheHit: the prefix is in the paged mxfp4 cache and the new
        # tokens' K/V were already quantized+scattered before this call, so the
        # cache holds the full (prefix+new) KV. Attend over it directly with
        # FIA v2 dequant (the op dequants fp4/e8m0 internally) instead of a
        # broken manual dequant. Mirrors upstream _get_fia_params PrefillCacheHit
        # block_table handling (slice to the real request count, no decode
        # offset since PrefillCacheHit only occurs with chunked_prefill off).
        num_block, block_size, _, _ = self.key_cache.shape
        key_bnsd = self.key_cache.view(num_block, block_size, -1)
        value_bnsd = self.value_cache.view(num_block, block_size, -1)
        k_scale = self.mxfp4_k_scale_cache.view(torch.float8_e8m0fnu)
        v_scale = self.mxfp4_v_scale_cache.view(torch.float8_e8m0fnu)
        # block_tables may be padded; slice to the real request count.
        batch_size = attn_metadata.seq_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]
        actual_seq_lengths_kv = attn_metadata.seq_lens_list
        query_rot = self._rotate(query)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=query_rot,
            key=key_bnsd,
            value=value_bnsd,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            softmax_scale=self.scale,
            **_mxfp4_fia_v2_dequant_kwargs(),
            sparse_mode=3,
        )

    attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
    output[:num_tokens] = attn_output
    return output


AscendAttentionBackendImpl._forward_mxfp4_fused_infer_attention = _forward_mxfp4_fused_infer_attention


def _mxfp4_update_graph_params(
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None
):
    is_mxfp4 = _is_mxfp4_kv_enabled()

    if not is_mxfp4:
        # Non-mxfp4 models: defer to the original implementation so this patch
        # does not regress vanilla bf16/fp8 graph replay.
        return _orig_update_graph_params(
            update_stream,
            forward_context,
            num_tokens,
            vllm_config,
            speculative_config,
            num_dcp_pcp_tokens,
            draft_attn_metadatas,
        )

    if _EXTRA_CTX.sinks:
        raise NotImplementedError(
            "mxfp4 KV cache does not yet support attention sinks"
        )
    else:
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
            attn_metadata = draft_attn_metadatas
            # Build (draft_step, key) pairs across all draft steps so captured
            # attn params replay against the right metadata, matching upstream
            # update_graph_params draft stepping.
            draft_attn_key_steps = [
                (draft_step, key)
                for draft_step, per_step_metadata in enumerate(attn_metadata)
                for key in per_step_metadata
            ]
            attn_keys = [key for _, key in draft_attn_key_steps]
        else:
            graph_params = get_graph_params()
            attn_metadata = forward_context.attn_metadata
            attn_keys = list(attn_metadata.keys())
            attn_keys_length = len(graph_params.attn_params[num_tokens])
            if attn_keys_length == 0:
                return
            # No sorting: each captured param carries its own layer name and
            # replay looks the metadata up by name (see metadata_key below),
            # so iteration order is irrelevant. This mirrors upstream
            # layer-aware replay (gemma4) and is robust to mixed-attention
            # models where metadata order != layer order.

        num_layers = len(attn_keys)
        if num_layers == 0:
            return
        graph_param_count = len(graph_params.attn_params[num_tokens])
        if _EXTRA_CTX.is_draft_model:
            # Align (draft_step, key) pairs to the captured param count,
            # repeating or truncating as needed (upstream uses cdiv here).
            if graph_param_count > len(draft_attn_key_steps):
                repeat_count = (
                    graph_param_count + len(draft_attn_key_steps) - 1
                ) // len(draft_attn_key_steps)
                draft_attn_key_steps = (
                    draft_attn_key_steps * repeat_count
                )[:graph_param_count]
            else:
                draft_attn_key_steps = draft_attn_key_steps[:graph_param_count]
            attn_keys = [key for _, key in draft_attn_key_steps]
        attn_count = 0
        with torch.npu.stream(update_stream):
            for key, param, handle, event in zip(
                attn_keys, 
                graph_params.attn_params[num_tokens],
                graph_params.handles[num_tokens],
                graph_params.events[num_tokens]
            ):
                (query, key_cache, value, block_tables, attn_mask,
                block_size, seq_lens, actual_seq_lengths_q,
                num_kv_heads, num_heads, scale,
                attn_output, softmax_lse,
                sparse_mode, pre_tokens, next_tokens,
                c8_k_aq_scale,
                c8_k_aq_offset,
                c8_v_aq_scale,
                c8_v_aq_offset,
                mxfp4_query_rot,
                mxfp4_orig_query,
                layer_name
                ) = param
                if _EXTRA_CTX.is_draft_model:
                    draft_step, key = draft_attn_key_steps[attn_count]
                    meta = attn_metadata[draft_step][key]
                    seq_lens = meta.seq_lens_list
                    actual_seq_lengths_q = meta.actual_seq_lengths_q
                    block_tables = meta.block_tables
                    attn_count += 1
                    if not meta.causal:
                        sparse_mode = 0
                else:
                    # Resolve metadata by the captured layer name (falls back
                    # to the zip key when the name is absent, mirroring
                    # upstream layer-aware replay).
                    metadata_key = (
                        layer_name
                        if layer_name is not None and layer_name in attn_metadata
                        else key
                    )
                    seq_lens = attn_metadata[metadata_key].seq_lens_list
                    actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q
                    # mxfp4 targets (Qwen3.5 / GLM5.2) have no sliding window,
                    # so block_tables always comes from live metadata.
                    block_tables = attn_metadata[metadata_key].block_tables
                
                if mxfp4_orig_query is not None and mxfp4_query_rot is not None:
                    _q_in = mxfp4_orig_query[:num_tokens]
                    Q = AscendAttentionBackendImpl._hadamard_32
                    if Q is None or Q.dtype != _q_in.dtype or Q.device != _q_in.device:
                        Q = _get_orthogonal_block(
                            device=_q_in.device, dtype=_q_in.dtype
                        )
                    _rotated = _q_in.reshape(-1, MXFP4_SCALE_GROUP_SIZE) @ Q
                    query.copy_(_rotated.reshape(_q_in.shape))
                
                k_scale_4d = c8_k_aq_scale.view(torch.float8_e8m0fnu)
                v_scale_4d = c8_v_aq_scale.view(torch.float8_e8m0fnu)

                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score_v2.out(
                    query=query,
                    key=key_cache,
                    value=value,
                    dequant_scale_key=k_scale_4d,
                    dequant_scale_value=v_scale_4d,
                    block_table=block_tables,
                    atten_mask=attn_mask,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_qlen=actual_seq_lengths_q,
                    actual_seq_kvlen=seq_lens,
                    num_key_value_heads=num_kv_heads,
                    num_query_heads=num_heads,
                    softmax_scale=scale,
                    **_mxfp4_fia_v2_dequant_kwargs(),
                    sparse_mode=sparse_mode,
                    workspace=graph_params.workspaces.get(num_tokens),
                    out=[attn_output, softmax_lse]
                )
                torch.npu.graph_task_update_end(update_stream)

                event.record(update_stream)

_orig_update_graph_params = AscendAttentionBackendImpl.update_graph_params

AscendAttentionBackendImpl.update_graph_params = staticmethod(
    _mxfp4_update_graph_params
)


def _precompute_mxfp4_workspaces(self):
    graph_params = get_graph_params()
    draft_graph_params = get_draft_graph_params()
    if graph_params is None and draft_graph_params is None:
        return

    # Collect all mxfp4 attn impls. Layers may have different shapes (e.g. mixed
    # attention types); take the max workspace across unique shapes, mirroring
    # upstream use_max_workspace, so a smaller layer's workspace isn't reused
    # for a larger one.
    unique_impls = {}
    for layer in self.compilation_config.static_forward_context.values():
        if not (hasattr(layer, "impl") and hasattr(layer.impl, "mxfp4_k_scale_cache")):
            continue
        impl = layer.impl
        if impl.key_cache is None or impl.mxfp4_k_scale_cache is None:
            continue
        shape_key = (impl.num_heads, impl.num_kv_heads, impl.head_size)
        unique_impls.setdefault(shape_key, impl)
    if not unique_impls:
        return

    dtype = self.dtype
    device = next(iter(unique_impls.values())).key_cache.device
    max_seq_len = self.vllm_config.model_config.max_model_len
    spec_config = self.vllm_config.speculative_config
    step = (1 + spec_config.num_speculative_tokens) if spec_config else 1

    def _ws_for_pattern(impl, num_tokens, batch_size, q_lens, sm, mask):
        num_block, block_size, _, _ = impl.key_cache.shape
        max_blocks_per_seq = max_seq_len // block_size
        key_3d = impl.key_cache.view(num_block, block_size, -1)
        value_3d = impl.value_cache.view(num_block, block_size, -1)
        ks_4d = impl.mxfp4_k_scale_cache.view(
            num_block, block_size, impl.num_kv_heads, -1)
        vs_4d = impl.mxfp4_v_scale_cache.view(
            num_block, block_size, impl.num_kv_heads, -1)
        bt = torch.zeros(
            batch_size, max_blocks_per_seq, dtype=torch.int32, device=device)
        kv_lens = torch.full(
            (batch_size,), max_seq_len, dtype=torch.int32, device=device)
        q = torch.zeros(
            num_tokens, impl.num_heads, impl.head_size,
            dtype=dtype, device=device)
        return torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
            query=q, key=key_3d, value=value_3d,
            dequant_scale_key=ks_4d.view(torch.float8_e8m0fnu),
            dequant_scale_value=vs_4d.view(torch.float8_e8m0fnu),
            block_table=bt, atten_mask=mask,
            input_layout="TND", block_size=block_size,
            actual_seq_qlen=q_lens, actual_seq_kvlen=kv_lens,
            num_key_value_heads=impl.num_kv_heads,
            num_query_heads=impl.num_heads,
            softmax_scale=impl.scale,
            **_mxfp4_fia_v2_dequant_kwargs(),
            sparse_mode=sm,
        )

    def _compute_workspace(num_tokens: int) -> torch.Tensor:
        pure_qlen = torch.arange(
            1, num_tokens + 1, dtype=torch.int32, device=device)
        best = None
        for impl in unique_impls.values():
            ws = _ws_for_pattern(
                impl, num_tokens, num_tokens, pure_qlen, 0, None)
            if best is None or ws.numel() > best.numel():
                best = ws
            if step > 1 and num_tokens % step == 0:
                mtp_batch = num_tokens // step
                mtp_qlen = torch.arange(
                    step, num_tokens + 1, step,
                    dtype=torch.int32, device=device)
                mtp_mask = torch.zeros(
                    2048, 2048, dtype=torch.bool, device=device)
                mtp_ws = _ws_for_pattern(
                    impl, num_tokens, mtp_batch, mtp_qlen, 3, mtp_mask)
                if mtp_ws.numel() > best.numel():
                    best = mtp_ws
        return best

    if graph_params is not None:
        for nt in list(graph_params.workspaces.keys()):
            if graph_params.workspaces[nt] is None:
                update_graph_params_workspaces(nt, _compute_workspace(nt))

    if draft_graph_params is not None:
        for nt in list(draft_graph_params.workspaces.keys()):
            if draft_graph_params.workspaces[nt] is None:
                main_ws = (
                    graph_params.workspaces.get(nt)
                    if graph_params is not None else None
                )
                if main_ws is not None:
                    update_draft_graph_params_workspaces(nt, main_ws)
                else:
                    update_draft_graph_params_workspaces(nt, _compute_workspace(nt))

NPUModelRunner._precompute_mxfp4_workspaces = _precompute_mxfp4_workspaces

_orig_reshape = NPUModelRunner._reshape_kv_cache_tensors


def _mxfp4_reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors):
    if not _is_mxfp4_kv_enabled():
        return _orig_reshape(self, kv_cache_config, kv_cache_raw_tensors)

    kv_caches = _orig_reshape(self, kv_cache_config, kv_cache_raw_tensors)

    for group in self._kv_cache_spec_attn_group_iterator():
        current_kv_cache_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue
            # Only C4 (MXFP4) layers are re-carved here; any other attention
            # spec (e.g. SlidingWindow in a mixed model) keeps the original
            # reshape result.
            if not isinstance(current_kv_cache_spec, AscendFullAttentionC4Spec):
                continue
            raw = kv_cache_raw_tensors.get(layer_name)

            # Get actual block layout from the standard reshape result.
            orig_result = kv_caches.get(layer_name)
            if not isinstance(orig_result, tuple) or len(orig_result) < 1:
                continue
            orig_k_cache = orig_result[0]
            actual_num_blocks = orig_k_cache.shape[0]
            actual_block_size = orig_k_cache.shape[1]

            hs_k = current_kv_cache_spec.head_size
            hs_v = getattr(current_kv_cache_spec, "head_size_v", hs_k) or hs_k
            sk = getattr(current_kv_cache_spec, "scale_dim", 0)
            sv = getattr(current_kv_cache_spec, "scale_dim_v", sk) or sk
            nk = current_kv_cache_spec.num_kv_heads

            k_data_numel = actual_num_blocks * actual_block_size * nk * hs_k
            k_scale_numel = actual_num_blocks * actual_block_size * nk * sk
            v_data_numel = actual_num_blocks * actual_block_size * nk * hs_v
            v_scale_numel = actual_num_blocks * actual_block_size * nk * sv

            if isinstance(raw, tuple) and len(raw) == 2:
                # Non-hybrid: separate K and V raw tensors.
                raw_k_tensor, raw_v_tensor = raw
                raw_k = raw_k_tensor.view(torch.uint8)
                raw_v = raw_v_tensor.view(torch.uint8)
                k_cache = raw_k[:k_data_numel].view(
                    actual_num_blocks, actual_block_size, nk, hs_k)
                k_scale_cache = raw_k[k_data_numel:k_data_numel + k_scale_numel].view(
                    actual_num_blocks, actual_block_size, nk, sk)
                v_cache = raw_v[:v_data_numel].view(
                    actual_num_blocks, actual_block_size, nk, hs_v)
                v_scale_cache = raw_v[v_data_numel:v_data_numel + v_scale_numel].view(
                    actual_num_blocks, actual_block_size, nk, sv)
            elif isinstance(raw, torch.Tensor):
                c4_total = k_data_numel + k_scale_numel + v_data_numel + v_scale_numel
                raw_u8 = raw.view(torch.uint8)
                base = raw_u8.numel() - c4_total
                k_cache = raw_u8[base:base + k_data_numel].view(
                    actual_num_blocks, actual_block_size, nk, hs_k)
                k_scale_cache = raw_u8[base + k_data_numel:base + k_data_numel + k_scale_numel].view(
                    actual_num_blocks, actual_block_size, nk, sk)
                v_base = base + k_data_numel + k_scale_numel
                v_cache = raw_u8[v_base:v_base + v_data_numel].view(
                    actual_num_blocks, actual_block_size, nk, hs_v)
                v_scale_cache = raw_u8[v_base + v_data_numel:v_base + v_data_numel + v_scale_numel].view(
                    actual_num_blocks, actual_block_size, nk, sv)
            else:
                continue

            kv_caches[layer_name] = (k_cache, v_cache, k_scale_cache, v_scale_cache)

    return kv_caches

NPUModelRunner._reshape_kv_cache_tensors = _mxfp4_reshape_kv_cache_tensors

_orig_get_kv_cache_spec = NPUModelRunner.get_kv_cache_spec


def _mxfp4_get_kv_cache_spec(self) -> dict:
    kv_cache_spec = _orig_get_kv_cache_spec(self)
    enabled = _is_mxfp4_kv_enabled()
    logger.info("[mxfp4_kv] _is_mxfp4_kv_enabled() = %s in get_kv_cache_spec", enabled)
    if not enabled:
        return kv_cache_spec
    c4_real_page_size = None
    replaced = 0
    for layer_name, spec in list(kv_cache_spec.items()):
        if isinstance(spec, FullAttentionSpec) and not isinstance(
            spec, AscendFullAttentionC4Spec
        ):
            kv_cache_spec[layer_name] = AscendFullAttentionC4Spec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=mxfp4_kv_cache_data_dim(spec.head_size),
                head_size_v=mxfp4_kv_cache_data_dim(spec.head_size_v),
                dtype=torch.uint8,
                kv_quant_mode=KVQuantMode.NONE,
                scale_dim=mxfp4_kv_cache_scale_dim(spec.head_size),
                scale_dim_v=mxfp4_kv_cache_scale_dim(spec.head_size_v),
                page_size_padded=None,
            )
            c4_real_page_size = kv_cache_spec[layer_name].real_page_size_bytes
            replaced += 1

    logger.info(
        "[mxfp4_kv] replaced %d dense attention spec(s) with AscendFullAttentionC4Spec",
        replaced,
    )
    if replaced == 0 or c4_real_page_size is None:
        return kv_cache_spec

    mamba_specs = [
        (n, s) for n, s in kv_cache_spec.items() if isinstance(s, MambaSpec)
    ]
    if mamba_specs:
        max_mamba_real = max(
            sum(math.prod(shape) * get_dtype_size(dtype)
                for shape, dtype in zip(s.shapes, s.dtypes))
            for _, s in mamba_specs
        )
        new_page_size = max_mamba_real + c4_real_page_size
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, (MambaSpec, AscendFullAttentionC4Spec)):
                continue
            other_ps = getattr(spec, "page_size_bytes", 0)
            if other_ps > new_page_size:
                new_page_size = other_ps

        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, (MambaSpec, AscendFullAttentionC4Spec)):
                object.__setattr__(spec, "page_size_padded", new_page_size)
            elif getattr(spec, "page_size_padded", None) is not None:
                object.__setattr__(spec, "page_size_padded", new_page_size)
    return kv_cache_spec


NPUModelRunner.get_kv_cache_spec = _mxfp4_get_kv_cache_spec

_orig_dummy_run = NPUModelRunner._dummy_run


def _mxfp4_dummy_run(self, *args, **kwargs):
    is_graph_capturing = kwargs.get("is_graph_capturing", False)
    if is_graph_capturing and _is_mxfp4_kv_enabled():
        self._precompute_mxfp4_workspaces()
    return _orig_dummy_run(self, *args, **kwargs)


NPUModelRunner._dummy_run = _mxfp4_dummy_run
