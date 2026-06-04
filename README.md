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

## CLI 示例

```powershell
ixsmi-vllm generate --prompt "Hello, my name is" --max-tokens 8
```

## 后续实现路线

- Step 1：完成 toy 后端与批处理推理（当前已实现）
- Step 2：接入真实 tokenizer / Hugging Face 模型（当前已实现，可选依赖 `[hf]`）
- Step 3：实现连续批处理（continuous batching）（当前已实现 step-wise scheduler）
- Step 4：实现 KV cache block manager / PagedAttention 数据结构（当前已实现 token-id block table，占位真实 KV tensor）
- Step 5：把 KVCacheManager 从 token-id block 升级为真实 K/V tensor block（当前已实现 `KVTensorRef` block table）
- Step 6：在 HuggingFaceBackend 或 CoreXBackend 中接入 `past_key_values`（当前已实现 HF `use_cache=True` / `past_key_values` 增量解码）
- Step 7：对接 BI-V150S corex.4.4.0 runtime kernels（当前已实现 `CoreXRuntimeAdapter` 协议边界；需厂商 SDK 后接入真实 kernel）
- Step 8：完善 OpenAI `/v1/completions`、`/v1/chat/completions`

> 说明：当前 `corex` 后端是插件化占位实现，会安全回退到 toy backend；等待具体 SDK/API 后即可在 `src/ixsmi_vllm/backends/corex_runtime.py` 中把 `init_state()` / `decode()` 映射到 BI-V150S corex.4.4.0 runtime kernels。
