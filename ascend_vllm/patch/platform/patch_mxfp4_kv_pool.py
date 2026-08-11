"""MXFP4 dual block pool: separate block pools for mamba and C4 attention.

Enabled via additional-config ``enable_mxfp4_kv: true`` on hybrid models that
mix ``MambaSpec`` (linear attention) and ``AscendFullAttentionC4Spec`` (MXFP4
full attention) layers.

Background: the single-pool layout pads every block to
``mamba_page + c4_page`` bytes so that one block id addresses both the mamba
state region and the attention KV region of a shared tensor. With a
linear:full = 3:1 model each cached prefix boundary needs 3 mamba blocks + 1
attention block, so only ``(3m + c) / (4(m + c))`` of the pool is ever useful
(~42% structural waste at m ~= 2c).

This patch gives each cache type its own block pool with independent block-id
spaces ("allocate separately, different layers may reuse the same block id"):

* mamba pool: ``n_m = ratio * n_a`` blocks of real ``mamba_page`` bytes
  (conv+ssm). The mamba groups keep the upstream cross-group tensor sharing
  (a block id is live in at most one group at a time).
* attention pool: ``n_a`` blocks of real ``c4_page`` bytes (data+scale).
  ``kv_cache_config.num_blocks == n_a`` so the "GPU KV cache size" log and
  the admission watermark reflect the attention pool.

Prefix caching: block hashes depend only on token content, so each pool keeps
its own hash -> block map. Mamba groups look up hits in the mamba pool,
attention groups in the main pool; the upstream fixed-point min reconciliation
in ``HybridKVCacheCoordinator.find_longest_cache_hit`` is unchanged, and a
prefix only counts as hit when BOTH pools cached it. Block tables are
per-group, so each group addressing its own pool's id space is correct.
Eviction is per-pool; divergence only ever shortens hits (via the min), never
mis-addresses.

Knob: additional-config ``mxfp4_mamba_pool_ratio`` (int >= 1, default 3).
``ratio`` blocks of mamba state are provisioned per attention-pool block;
with 3 mamba groups sharing the mamba pool, ratio=3 lets both pools cache
about the same number of prefix boundaries. The ratio rides on the mamba
group specs (see MXFP4_MAMBA_POOL_RATIO_ATTR) instead of an absolute block
count so the cross-worker min-num_blocks proportional shrink in
``get_kv_cache_configs`` scales both pools consistently (mamba tensors are
sized ``n_a * ratio * mamba_page``, divisible by ``n_a``).

Patched upstream functions (all resolved through module globals at call
time, so no call-site patching is needed):

* ``kv_cache_utils.get_kv_cache_groups``: skip page-size unification; the
  bucketing helper only needs spec equality, not uniform pages.
* ``kv_cache_utils.get_kv_cache_config_from_groups``: per-type tensors.
* ``kv_cache_utils._pool_bytes_per_block``: dual-pool bytes per n_a unit
  (used when num_gpu_blocks_override is set).
* ``kv_cache_utils.get_kv_cache_capacity``: min over the two pools.
* ``KVCacheCoordinator.__init__``: create the mamba BlockPool and hand it to
  the mamba single-type managers (cache/touch/allocate/free all go through
  the manager's own pool).
* ``KVCacheCoordinator.get_num_blocks_to_allocate``: per-pool admission
  accounting (mamba demand against the mamba pool; the returned value charges
  only the attention demand against the main pool).
* ``MambaManager.find_longest_cache_hit``: look up hits in the mamba pool.

Limitations: requires the hybrid KV cache manager, an explicit max_model_len
(no auto-fit), and only MambaSpec + AscendFullAttentionC4Spec groups.
"""

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    MambaManager,
)

# vllm-ascend 上多 group + caching 实际实例化的是 AscendHybridKVCacheCoordinator，
# 其 __init__ 完全重写且不调用 super().__init__()，自建主 BlockPool 并把所有
# manager 挂上去 —— 因此只 patch KVCacheCoordinator.__init__ 是死代码，双池永远
# 不会生效（mamba 前缀命中恒为 0）。这里 patch AscendHybridKVCacheCoordinator；
# 若该模块不可用（版本差异）则回退到 KVCacheCoordinator。
try:
    from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
        AscendHybridKVCacheCoordinator,
    )
    _ASCEND_HYBRID_COORDINATOR = True
except ImportError:  # pragma: no cover
    AscendHybridKVCacheCoordinator = None
    _ASCEND_HYBRID_COORDINATOR = False
from vllm.v1.kv_cache_interface import (
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheTensor,
    MambaSpec,
)

from ascend_vllm.patch.platform.patch_mxfp4_cache_config import (
    MXFP4_MAMBA_POOL_RATIO_ATTR,
    MXFP4_MAMBA_POOL_RATIO_DEFAULT,
    AscendFullAttentionC4Spec,
    get_mxfp4_mamba_pool_ratio,
)

logger = init_logger(__name__)

# Sentinel returned by the admission wrapper when the mamba pool cannot fit
# the request: large enough to fail every `required > free` check.
_ADMISSION_REJECT = 2**62


def _mxfp4_kv_enabled(vllm_config) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return bool(additional_config.get("enable_mxfp4_kv", False))


def _auto_mamba_pool_ratio(kv_cache_groups) -> int:
    """Derive the mamba:attention pool ratio from the layer counts.

    Prefix caching needs the mamba pool to hold roughly as many prefix states
    as the attention pool: one prefix token consumes one mamba block per mamba
    group and one C4 block per attention group, so the ratio is the group-count
    ratio. Hard-coding 3 only fits a 3:1 layer layout.
    """
    num_mamba = sum(
        len(g.layer_names)
        for g in kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
    )
    num_c4 = sum(
        len(g.layer_names)
        for g in kv_cache_groups
        if isinstance(g.kv_cache_spec, AscendFullAttentionC4Spec)
    )
    if num_c4 == 0:
        return MXFP4_MAMBA_POOL_RATIO_DEFAULT
    return max(1, cdiv(num_mamba, num_c4))


def _mamba_pool_ratio(vllm_config, kv_cache_groups=None) -> int:
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    configured = additional_config.get("mxfp4_mamba_pool_ratio", None)
    if configured is not None:
        ratio = int(configured)
        if ratio < 1:
            raise ValueError(
                f"mxfp4_mamba_pool_ratio must be >= 1, got {ratio}"
            )
        return ratio
    if kv_cache_groups is not None:
        return _auto_mamba_pool_ratio(kv_cache_groups)
    return MXFP4_MAMBA_POOL_RATIO_DEFAULT


def _split_dual_pool_groups(kv_cache_groups):
    """Split groups into (mamba_groups, c4_groups); None if not a pure
    mamba+C4 dual-pool layout."""
    mamba_groups = [
        g for g in kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    c4_groups = [
        g
        for g in kv_cache_groups
        if isinstance(g.kv_cache_spec, AscendFullAttentionC4Spec)
    ]
    if not mamba_groups or not c4_groups:
        return None
    if len(mamba_groups) + len(c4_groups) != len(kv_cache_groups):
        return None
    return mamba_groups, c4_groups


def _dual_pool_ratio_from_config(kv_cache_config) -> int | None:
    for group in kv_cache_config.kv_cache_groups:
        if isinstance(group.kv_cache_spec, MambaSpec):
            ratio = get_mxfp4_mamba_pool_ratio(group.kv_cache_spec)
            if ratio is not None:
                return ratio
    return None


# ---------------------------------------------------------------------------
# 1. Grouping: skip page-size unification for dual-pool models.
# ---------------------------------------------------------------------------

_orig_get_kv_cache_groups = kv_cache_utils.get_kv_cache_groups


def _patched_get_kv_cache_groups(vllm_config, kv_cache_spec):
    has_mamba = any(isinstance(s, MambaSpec) for s in kv_cache_spec.values())
    has_c4 = any(
        isinstance(s, AscendFullAttentionC4Spec) for s in kv_cache_spec.values()
    )
    if not (_mxfp4_kv_enabled(vllm_config) and has_mamba and has_c4):
        return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)

    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError(
            "enable_mxfp4_kv (dual block pool) requires the hybrid KV cache "
            "manager; do not set --disable-hybrid-kv-cache-manager."
        )
    if vllm_config.model_config.original_max_model_len == -1:
        raise ValueError(
            "enable_mxfp4_kv (dual block pool) requires an explicit "
            "max_model_len; auto-fit max_model_len is not supported."
        )
    if any(isinstance(s, HiddenStateCacheSpec) for s in kv_cache_spec.values()):
        raise ValueError(
            "enable_mxfp4_kv (dual block pool) does not support "
            "HiddenStateCacheSpec layers."
        )

    # Skip unify_kv_cache_spec_page_size: mamba (conv+ssm) and C4 (data+scale)
    # pages intentionally differ under dual pool. The bucketing helper only
    # requires spec equality, not uniform pages.
    return kv_cache_utils._get_kv_cache_groups_uniform_page_size(kv_cache_spec)


kv_cache_utils.get_kv_cache_groups = _patched_get_kv_cache_groups


# ---------------------------------------------------------------------------
# 2. Config builder: per-type tensors and pool sizes.
# ---------------------------------------------------------------------------

_orig_get_kv_cache_config_from_groups = kv_cache_utils.get_kv_cache_config_from_groups


def _patched_get_kv_cache_config_from_groups(
    vllm_config, kv_cache_groups, available_memory
):
    split = _split_dual_pool_groups(kv_cache_groups)
    if split is None or not _mxfp4_kv_enabled(vllm_config):
        return _orig_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )
    mamba_groups, c4_groups = split

    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    mamba_spec = mamba_groups[0].kv_cache_spec
    c4_spec = c4_groups[0].kv_cache_spec
    m_page = mamba_spec.page_size_bytes
    c_page = c4_spec.page_size_bytes
    assert all(g.kv_cache_spec.page_size_bytes == m_page for g in mamba_groups)
    assert all(g.kv_cache_spec.page_size_bytes == c_page for g in c4_groups)

    ratio = _mamba_pool_ratio(vllm_config, kv_cache_groups)
    _ratio_cfg = (
        getattr(vllm_config, "additional_config", None) or {}
    ).get("mxfp4_mamba_pool_ratio", None)
    logger.info(
        "[mxfp4_kv] mamba pool ratio=%d (%s)",
        ratio,
        f"configured mxfp4_mamba_pool_ratio={_ratio_cfg}"
        if _ratio_cfg is not None
        else (
            f"auto-derived from {sum(len(g.layer_names) for g in mamba_groups)} "
            f"mamba / {sum(len(g.layer_names) for g in c4_groups)} c4 layers"
        ),
    )
    # One unit of num_blocks costs ratio mamba pages + 1 C4 page per
    # group-size column.
    bytes_per_unit = group_size * (ratio * m_page + c_page)
    n_a = kv_cache_utils.may_override_num_blocks(
        vllm_config, available_memory // bytes_per_unit
    )
    if n_a <= 0:
        raise ValueError(
            "Not enough KV cache memory for the mxfp4 dual block pool: "
            f"available_memory={available_memory}, one unit needs "
            f"{bytes_per_unit} bytes (group_size={group_size}, ratio={ratio}, "
            f"mamba_page={m_page}, c4_page={c_page})."
        )

    # The mamba pool must also hold the resident state of a fully occupied
    # batch (align mode keeps 2 + num_speculative_blocks state blocks per
    # request per group). Round the resident floor up to a multiple of n_a so
    # the mamba tensor (n_m * m_page) stays divisible by num_blocks and the
    # cross-worker proportional shrink in get_kv_cache_configs still works.
    per_req_blocks = mamba_spec.max_memory_usage_bytes(vllm_config) // m_page
    resident_min = (
        len(mamba_groups)
        * vllm_config.scheduler_config.max_num_seqs
        * per_req_blocks
    )
    n_m = max(ratio * n_a, cdiv(resident_min, n_a) * n_a)
    if n_m > ratio * n_a:
        if n_m * m_page * group_size + n_a * c_page * group_size > available_memory:
            # Resident floor would exceed the budget; fall back to the pure
            # ratio sizing and let admission throttle before starvation.
            n_m = ratio * n_a
        logger.warning(
            "[mxfp4_kv] mamba pool sized to %d blocks (resident minimum %d "
            "for max_num_seqs=%d, ratio=%d); attention pool capacity may be "
            "reduced.",
            n_m,
            resident_min,
            vllm_config.scheduler_config.max_num_seqs,
            ratio,
        )

    kv_cache_tensors: list[KVCacheTensor] = []
    for i in range(group_size):
        # Mamba column: one tensor per position, shared across the mamba
        # groups at that position (cross-group block-id exclusivity, same as
        # upstream). Sized n_m * m_page so the cross-worker proportional
        # shrink scales it with num_blocks.
        mamba_shared = [
            g.layer_names[i] for g in mamba_groups if i < len(g.layer_names)
        ]
        if mamba_shared:
            kv_cache_tensors.append(
                KVCacheTensor(size=n_m * m_page, shared_by=mamba_shared)
            )
        c4_shared = [
            g.layer_names[i] for g in c4_groups if i < len(g.layer_names)
        ]
        if c4_shared:
            kv_cache_tensors.append(
                KVCacheTensor(size=n_a * c_page, shared_by=c4_shared)
            )

    # Carry the pool ratio on the mamba group specs so the scheduler-side
    # coordinator patch and the worker-side reshape patch can size the mamba
    # pool as ratio * num_blocks after the cross-worker min shrink.
    for g in mamba_groups:
        object.__setattr__(g.kv_cache_spec, MXFP4_MAMBA_POOL_RATIO_ATTR, ratio)

    logger.info(
        "[mxfp4_kv] dual block pool: attention pool %d blocks x %d B, "
        "mamba pool %d blocks x %d B (ratio=%d, %d mamba groups, %d C4 "
        "groups, group_size=%d)",
        n_a,
        c_page,
        n_m,
        m_page,
        ratio,
        len(mamba_groups),
        len(c4_groups),
        group_size,
    )
    return KVCacheConfig(
        num_blocks=n_a,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


kv_cache_utils.get_kv_cache_config_from_groups = (
    _patched_get_kv_cache_config_from_groups
)


# ---------------------------------------------------------------------------
# 3. num_gpu_blocks_override support: bytes per num_blocks unit.
# ---------------------------------------------------------------------------

_orig_pool_bytes_per_block = kv_cache_utils._pool_bytes_per_block


def _patched_pool_bytes_per_block(vllm_config, kv_cache_groups) -> int:
    split = _split_dual_pool_groups(kv_cache_groups)
    if split is None or not _mxfp4_kv_enabled(vllm_config):
        return _orig_pool_bytes_per_block(vllm_config, kv_cache_groups)
    mamba_groups, c4_groups = split
    group_size = max(len(g.layer_names) for g in kv_cache_groups)
    ratio = _mamba_pool_ratio(vllm_config, kv_cache_groups)
    return group_size * (
        ratio * mamba_groups[0].kv_cache_spec.page_size_bytes
        + c4_groups[0].kv_cache_spec.page_size_bytes
    )


kv_cache_utils._pool_bytes_per_block = _patched_pool_bytes_per_block


# ---------------------------------------------------------------------------
# 4. Capacity logging: min over the two pools.
# ---------------------------------------------------------------------------

_orig_get_kv_cache_capacity = kv_cache_utils.get_kv_cache_capacity


def _patched_get_kv_cache_capacity(vllm_config, kv_cache_config):
    ratio = _dual_pool_ratio_from_config(kv_cache_config)
    if ratio is None:
        return _orig_get_kv_cache_capacity(vllm_config, kv_cache_config)

    def _blocks_per_request(groups) -> int:
        return sum(
            cdiv(
                g.kv_cache_spec.max_memory_usage_bytes(vllm_config),
                g.kv_cache_spec.page_size_bytes,
            )
            for g in groups
        )

    mamba_groups = [
        g
        for g in kv_cache_config.kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    attn_groups = [
        g
        for g in kv_cache_config.kv_cache_groups
        if not isinstance(g.kv_cache_spec, MambaSpec)
    ]
    n_a = kv_cache_config.num_blocks
    candidates = []
    attn_bpr = _blocks_per_request(attn_groups)
    mamba_bpr = _blocks_per_request(mamba_groups)
    if attn_bpr > 0:
        candidates.append(n_a / attn_bpr)
    if mamba_bpr > 0:
        candidates.append(ratio * n_a / mamba_bpr)
    max_concurrency = min(candidates) if candidates else 0.0
    max_model_len = vllm_config.model_config.max_model_len
    logger.info(
        "[mxfp4_kv] capacity: attention pool=%d blocks (%d blocks/req), "
        "mamba pool=%d blocks (%d blocks/req), max_concurrency=%.2f, "
        "kv_cache_size_tokens=%d",
        n_a,
        attn_bpr,
        ratio * n_a,
        mamba_bpr,
        max_concurrency,
        int(max_concurrency * max_model_len),
    )
    return int(max_concurrency * max_model_len), max_concurrency


kv_cache_utils.get_kv_cache_capacity = _patched_get_kv_cache_capacity


# ---------------------------------------------------------------------------
# 5. Coordinator: a second BlockPool for the mamba groups.
# ---------------------------------------------------------------------------

# id(main block pool) -> mamba block pool, so the MambaManager cache-hit
# lookup (which receives the main pool from the coordinator) can substitute
# the mamba pool.
_MAMBA_POOLS: dict[int, BlockPool] = {}

_coordinator_init_target = (
    AscendHybridKVCacheCoordinator
    if _ASCEND_HYBRID_COORDINATOR
    else KVCacheCoordinator
)
_orig_coordinator_init = _coordinator_init_target.__init__


def _patched_coordinator_init(self, *args, **kwargs):
    _orig_coordinator_init(self, *args, **kwargs)
    # kv_cache_config 在 orig init 里已挂到实例（两种 coordinator 都如此）。
    kv_cache_config = self.kv_cache_config
    ratio = _dual_pool_ratio_from_config(kv_cache_config)
    if ratio is None:
        return
    logger.info(
        "[mxfp4_kv] activating dual block pool via %s (ratio=%d, attention "
        "pool=%d blocks); if this line is missing, the coordinator class "
        "changed and dual pool is NOT active",
        _coordinator_init_target.__name__,
        ratio,
        kv_cache_config.num_blocks,
    )
    n_m = ratio * kv_cache_config.num_blocks
    mamba_pool = BlockPool(
        num_gpu_blocks=n_m,
        enable_caching=self.enable_caching,
        hash_block_size=self.block_pool.hash_block_size,
        # KV cache events carry bare block ids, which collide across pools;
        # mamba state is local-only, so disable events on the mamba pool.
        enable_kv_cache_events=False,
        metrics_collector=None,
    )
    self.mamba_block_pool = mamba_pool
    _MAMBA_POOLS[id(self.block_pool)] = mamba_pool
    swapped = 0
    for group, manager in zip(
        kv_cache_config.kv_cache_groups, self.single_type_managers
    ):
        if isinstance(group.kv_cache_spec, MambaSpec):
            manager.block_pool = mamba_pool
            swapped += 1
    logger.info(
        "[mxfp4_kv] mamba block pool: %d blocks (ratio=%d x attention pool "
        "%d), %d mamba manager(s) detached from the main pool",
        n_m,
        ratio,
        kv_cache_config.num_blocks,
        swapped,
    )


_coordinator_init_target.__init__ = _patched_coordinator_init


# ---------------------------------------------------------------------------
# 6. Cache-hit lookup: mamba groups query the mamba pool.
# ---------------------------------------------------------------------------

_orig_mamba_find_longest_cache_hit = MambaManager.find_longest_cache_hit


def _patched_mamba_find_longest_cache_hit(
    cls, block_hashes, max_length, kv_cache_group_ids, block_pool, *args, **kwargs
):
    mamba_pool = _MAMBA_POOLS.get(id(block_pool))
    if mamba_pool is not None:
        block_pool = mamba_pool
        logger.debug(
            "[mxfp4_kv] mamba cache-hit lookup routed to mamba pool "
            "(%d blocks); hits>0 means dual-pool prefix caching is active",
            mamba_pool.num_gpu_blocks,
        )
    return _orig_mamba_find_longest_cache_hit(
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        *args,
        **kwargs,
    )


MambaManager.find_longest_cache_hit = classmethod(
    _patched_mamba_find_longest_cache_hit
)


# ---------------------------------------------------------------------------
# 7. Admission: per-pool capacity accounting.
# ---------------------------------------------------------------------------

_orig_get_num_blocks_to_allocate = KVCacheCoordinator.get_num_blocks_to_allocate


def _patched_get_num_blocks_to_allocate(
    self,
    request_id,
    num_tokens,
    new_computed_blocks,
    num_encoder_tokens,
    total_computed_tokens,
    num_local_computed_tokens,
    num_tokens_main_model,
    apply_admission_cap=False,
):
    mamba_pool = getattr(self, "mamba_block_pool", None)
    if mamba_pool is None:
        return _orig_get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            num_encoder_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )

    attn_demand = 0
    mamba_demand = 0
    for i, manager in enumerate(self.single_type_managers):
        if isinstance(manager, CrossAttentionManager):
            attn_demand += manager.get_num_blocks_to_allocate(
                request_id,
                num_encoder_tokens,
                [],
                0,
                0,
                num_encoder_tokens,
                apply_admission_cap=apply_admission_cap,
            )
            continue
        demand = manager.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks[i],
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )
        if isinstance(manager, MambaManager):
            mamba_demand += demand
        else:
            attn_demand += demand

    if mamba_demand > mamba_pool.get_num_free_blocks():
        # The mamba pool cannot fit this request now; fail the capacity
        # checks in allocate_slots so the request stays waiting/preempted.
        # Note MambaManager itself returns pool_size + 1 to defer a request
        # whose hit blocks were cached this step, which lands here too.
        return _ADMISSION_REJECT
    # Only the attention demand is charged against the main (attention)
    # pool; the mamba demand was validated against the mamba pool above.
    return attn_demand


KVCacheCoordinator.get_num_blocks_to_allocate = (
    _patched_get_num_blocks_to_allocate
)
