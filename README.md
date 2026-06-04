# ixsmi-vLLM

一个面向 **天数智凯 BI-V150S / corex.4.4.0** 的 Python 教学型 vLLM 风格推理框架。

首版目标不是复刻完整 vLLM，而是把核心链路分步实现清楚：

1. `LLM.generate()` 离线批量推理入口
2. `SamplingParams` 采样参数
3. 请求调度器与输出结构
4. 简单 tokenizer 与 toy 模型后端
5. BI-V150S/corex 后端适配接口占位
6. OpenAI-compatible server 的最小可选入口

## 快速运行

```powershell
python -m pip install -e .[dev]
python examples/offline_generate.py
pytest
```

如需接入真实 Hugging Face tokenizer / causal LM：

```powershell
python -m pip install -e .[hf]
ixsmi-vllm generate --backend hf --model sshleifer/tiny-gpt2 --prompt "Hello, my name is" --max-tokens 16
```

> `[hf]` 会安装 `transformers`、`torch` 和 `accelerate`。如果手动安装依赖且缺少 `accelerate`，框架会自动从 `device_map="auto"` 回退到普通 `cuda`/`cpu` 加载。

如果机器 NVIDIA 驱动过旧，PyTorch 可能会提示 CUDA 初始化警告；可先强制使用 CPU 验证功能：

```powershell
ixsmi-vllm generate --backend hf --device cpu --model sshleifer/tiny-gpt2 --prompt "Hello, my name is" --max-tokens 16
```

## CLI 示例

```powershell
ixsmi-vllm generate --prompt "Hello, my name is" --max-tokens 8
```

## BI-V150S corex.4.4.0 设备检测

在 BI-V150S 机器上可先执行驱动初始化和 16 卡可见性检测：

```powershell
ixsmi-vllm corex-info --corex-root /usr/local/corex-4.4.0 --visible-devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

该命令会设置 `IX_VISIBLE_DEVICES`、`ILUVATAR_COREX_ROOT`、`LD_LIBRARY_PATH`、`VLLM_TARGET_DEVICE`、`CUDA_VISIBLE_DEVICES`、`CUDA_HOME`，加载 `libixthunk.so` / `libcuda.so.1`，并调用 CUDA Driver API 的 `cuInit()` 与 `cuDeviceGetCount()`。

## OpenAI-compatible API

安装服务依赖并启动：

```powershell
python -m pip install -e .[server]
ixsmi-vllm-server --host 0.0.0.0 --port 8000 --model toy --backend toy
```

Completions API：

```powershell
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"model":"toy","prompt":"Hello, my name is","max_tokens":8,"temperature":0}'
```

Chat Completions API：

```powershell
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"toy","messages":[{"role":"user","content":"Say hello"}],"max_tokens":8,"temperature":0}'
```

## 后续实现路线

- Step 1：完成 toy 后端与批处理推理（当前已实现）
- Step 2：接入真实 tokenizer / Hugging Face 模型（当前已实现，可选依赖 `[hf]`）
- Step 3：实现连续批处理（continuous batching）（当前已实现 step-wise scheduler）
- Step 4：实现 KV cache block manager / PagedAttention 数据结构（当前已实现 token-id block table，占位真实 KV tensor）
- Step 5：把 KVCacheManager 从 token-id block 升级为真实 K/V tensor block（当前已实现 `KVTensorRef` block table）
- Step 6：在 HuggingFaceBackend 或 CoreXBackend 中接入 `past_key_values`（当前已实现 HF `use_cache=True` / `past_key_values` 增量解码）
- Step 7：对接 BI-V150S corex.4.4.0 runtime kernels（当前已实现 corex 驱动初始化、动态库加载和设备计数；模型 prefill/decode kernel 需厂商 SDK API 后映射）
- Step 8：完善 OpenAI `/v1/completions`、`/v1/chat/completions`（当前已实现 OpenAI 风格 `id/object/created/model/choices/usage` 响应）

> 说明：当前 `corex` 后端是插件化实现；无 runtime 注入时会安全回退到 toy backend。`src/ixsmi_vllm/backends/corex_runtime.py` 已接入 corex.4.4.0 驱动初始化与设备检测；等待模型执行 SDK/API 后，即可把 `init_state()` / `decode()` 映射到 BI-V150S prefill/decode kernels。
