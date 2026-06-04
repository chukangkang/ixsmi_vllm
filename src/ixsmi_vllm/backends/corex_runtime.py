from __future__ import annotations

import ctypes
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

from ixsmi_vllm.backends.base import DecodeResult, DecodeState, KVTensorRef


@dataclass
class CoreXCacheHandle:
    """Backend-specific handle for BI-V150S corex KV cache buffers."""

    model: str
    handle: object | None = None
    cache_length: int = 0


@dataclass(frozen=True)
class CoreXDeviceInfo:
    initialized: bool
    device_count: int
    error_code: int
    corex_root: str
    visible_devices: str


class CoreXRuntimeAdapter:
    """Adapter boundary for BI-V150S corex.4.4.0 runtime kernels.

    The concrete vendor Python package/API is not available in this workspace, so
    this class does not pretend to execute kernels. Once the SDK is installed,
    wire these methods to runtime calls such as model loading, prefill/decode,
    and KV cache block allocation.
    """

    candidate_modules = ("corex", "corex_runtime", "ixsmi_corex")

    def __init__(
        self,
        *,
        corex_root: str | None = None,
        visible_devices: str | None = None,
        load_libraries: bool = True,
    ) -> None:
        self.corex_root = corex_root or os.environ.get(
            "ILUVATAR_COREX_ROOT",
            "/usr/local/corex-4.4.0",
        )
        self.visible_devices = visible_devices or os.environ.get(
            "IX_VISIBLE_DEVICES",
            "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        )
        self.libcuda = None
        self.libixthunk = None
        self.configure_environment()
        if load_libraries:
            self.load_driver_libraries()
        self.runtime = self._load_runtime_module()

    def configure_environment(self) -> None:
        corex_root = Path(self.corex_root)
        os.environ["IX_VISIBLE_DEVICES"] = self.visible_devices
        os.environ["ILUVATAR_COREX_ROOT"] = str(corex_root)
        os.environ["VLLM_TARGET_DEVICE"] = "iluvatar"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.visible_devices
        os.environ["CUDA_HOME"] = str(corex_root)
        self._prepend_env_path("LD_LIBRARY_PATH", str(corex_root / "lib64"))
        self._prepend_env_path("PATH", str(corex_root / "bin"))

    def load_driver_libraries(self) -> None:
        lib_dir = Path(self.corex_root) / "lib64"
        self.libixthunk = ctypes.CDLL(
            str(lib_dir / "libixthunk.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.libcuda = ctypes.CDLL(
            str(lib_dir / "libcuda.so.1"),
            mode=ctypes.RTLD_GLOBAL,
        )

    def device_info(self) -> CoreXDeviceInfo:
        if self.libcuda is None:
            self.load_driver_libraries()

        cu_init = self.libcuda.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int

        cu_device_get_count = self.libcuda.cuDeviceGetCount
        cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu_device_get_count.restype = ctypes.c_int

        err = int(cu_init(0))
        count = ctypes.c_int(0)
        if err == 0:
            cu_device_get_count(ctypes.byref(count))

        return CoreXDeviceInfo(
            initialized=err == 0,
            device_count=int(count.value),
            error_code=err,
            corex_root=self.corex_root,
            visible_devices=self.visible_devices,
        )

    def init_state(self, model: str) -> CoreXCacheHandle:
        return CoreXCacheHandle(model=model)

    def decode(self, token_ids: list[int], state: object) -> DecodeResult:
        if not isinstance(state, CoreXCacheHandle):
            state = CoreXCacheHandle(model="unknown", handle=state)
        raise NotImplementedError(
            "CoreX driver initialization is wired. Model prefill/decode kernels "
            "still need vendor SDK calls. Map this method to corex.4.4.0 APIs "
            "and return DecodeResult(logits, state, kv_tensors)."
        )

    def make_kv_ref(self, layer_index: int, key_buffer: object, value_buffer: object) -> KVTensorRef:
        return KVTensorRef(layer_index=layer_index, key=key_buffer, value=value_buffer)

    def _load_runtime_module(self) -> object | None:
        for module_name in self.candidate_modules:
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
        return None

    def _prepend_env_path(self, name: str, value: str) -> None:
        current = os.environ.get(name, "")
        parts = [part for part in current.split(os.pathsep) if part]
        if value not in parts:
            os.environ[name] = value if not current else value + os.pathsep + current
