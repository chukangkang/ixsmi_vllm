from __future__ import annotations

import ctypes

from ixsmi_vllm.backends.corex_runtime import CoreXRuntimeAdapter


class FakeCFunction:
    def __init__(self, func):
        self.func = func
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.func(*args)


class FakeCudaLibrary:
    def __init__(self, count: int = 16, init_error: int = 0) -> None:
        self._count = count
        self.cuInit = FakeCFunction(lambda flags: init_error)
        self.cuDeviceGetCount = FakeCFunction(self._device_count)

    def _device_count(self, count_ptr) -> int:
        count_ptr._obj.value = self._count
        return 0


def test_corex_runtime_configures_16_card_environment(monkeypatch) -> None:
    runtime = CoreXRuntimeAdapter(load_libraries=False)

    assert runtime.visible_devices == "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    assert runtime.corex_root == "/usr/local/corex-4.4.0"
    assert "IX_VISIBLE_DEVICES" in __import__("os").environ
    assert __import__("os").environ["VLLM_TARGET_DEVICE"] == "iluvatar"


def test_corex_runtime_device_info_success(monkeypatch) -> None:
    loaded_paths: list[str] = []

    def fake_cdll(path, mode):
        loaded_paths.append(path)
        if path.endswith("libcuda.so.1"):
            return FakeCudaLibrary(count=16)
        return object()

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    runtime = CoreXRuntimeAdapter(load_libraries=False)

    info = runtime.device_info()

    assert info.initialized is True
    assert info.device_count == 16
    assert info.error_code == 0
    assert any(path.endswith("libixthunk.so") for path in loaded_paths)
    assert any(path.endswith("libcuda.so.1") for path in loaded_paths)
