#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.abc
import sys

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply vllm-ascend's process-wide patches once per process."""
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from ascend_vllm.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the PATCH NPU platform."""
    return "ascend_vllm.platform.PatchNPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    _ensure_global_patch()

    from vllm_ascend.model_loader.netloader import register_netloader
    from vllm_ascend.model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from vllm_ascend.profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from vllm_ascend.models import register_model

    register_model()


def register_kv_failure_patch():
    """Load Mooncake Hybrid KV failure patch in vLLM general-plugin processes."""
    from ascend_vllm.patch.platform import patch_recompute_scheduler  # noqa: F401
    from ascend_vllm.patch.worker import patch_mooncake_hybrid_connector  # noqa: F401


def register_general_plugin_patch():
    """Load ModelArts runtime patches through vLLM general plugins."""
    register_kv_failure_patch()
    # Explicitly load worker patches too (patch_mxfp4 etc.); this runs in
    # every process where general plugins load (engine core, API server,
    # worker processes), complementing the meta-path hook for processes
    # where general plugins are filtered out by VLLM_PLUGINS.
    from ascend_vllm.patch import worker  # noqa: F401


# ---------------------------------------------------------------------------
# Meta-path import hooks for reliable patch loading.
#
# Following the ascend-vllm (v6.5.306) pattern: hooks are installed at
# ``ascend_vllm`` import time (when vllm resolves the platform plugin, which
# happens in every process) and intercept imports that occur naturally during
# vllm startup, then trigger the matching ModelArts patch package:
#
# * ``vllm_ascend.ops``                       → ``ascend_vllm.patch.platform``
# * ``vllm_ascend.worker.model_runner_v1``    → ``ascend_vllm.patch.worker``
# ---------------------------------------------------------------------------


class _PatchImportLoader(importlib.abc.Loader):
    """Wrap the original loader; run ``on_loaded`` once after exec_module."""

    def __init__(self, original, on_loaded):
        self._original = original
        self._on_loaded = on_loaded
        self._done = False

    def create_module(self, spec):
        return self._original.create_module(spec)

    def exec_module(self, module):
        self._original.exec_module(module)
        if not self._done:
            self._done = True
            self._on_loaded()


class _PatchImportHook(importlib.abc.MetaPathFinder):
    """Intercept the import of one ``target`` and run ``on_loaded`` once."""

    def __init__(self, target, on_loaded):
        self._target = target
        self._on_loaded = on_loaded
        self._done = False

    def find_spec(self, name, path, target=None):
        if name == self._target and not self._done:
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = f.find_spec(name, path, target)
                if spec is not None:
                    spec.loader = _PatchImportLoader(spec.loader, self._on_loaded)
                    return spec
        return None


def _install_patch_import_hook(target, on_loaded):
    if not any(
        isinstance(f, _PatchImportHook) and f._target == target
        for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _PatchImportHook(target, on_loaded))
    if target in sys.modules:
        # Target already finished importing (we were imported late): apply
        # the patch set right away instead of waiting for a new import.
        on_loaded()


def _load_platform_patches():
    import ascend_vllm.patch.platform  # noqa: F401


def _load_worker_patches():
    import ascend_vllm.patch.worker  # noqa: F401


_install_patch_import_hook("vllm_ascend.ops", _load_platform_patches)
_install_patch_import_hook("vllm_ascend.worker.model_runner_v1", _load_worker_patches)
