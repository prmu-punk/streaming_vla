# rtc_async 模块文档

`model/rtc_async` 是一套与原有 `model/vla_qwen3.py` 并行的重写路径，目标是支持三件事：
- 训练时 RTC（action inpainting / delay masking）
- 推理时异步调度（`inference_delay` + `execute_horizon`）
- VLM KV-cache 与连续动作扩散头的直接互联（绕过 OAT detokenizer）

## 文件树

```text
model/rtc_async/
  README.md
  __init__.py
  action_expert/
    __init__.py
    diffusion_head.py
    model.py
    runner.py
  configs/
    rtc_async_vla.yaml
  pipeline/
    __init__.py
    scheduler.py
  qwen3_stream/
    __init__.py
    kv_export.py
    stream_runner_snapshot.py
  training/
    __init__.py
    loss_rtc.py

configs/
  vla_qwen3_rtc.yaml  # RTC 专用 VLM 条件编码配置（位于仓库根 configs/）
```

## 对外接口

- 顶层聚合导出：`model.rtc_async.__init__`
  - 动作专家：`ActionExpertRunner`、`ActionExpertBackbone`
  - 调度：`RTCChunkScheduler`
  - 训练 RTC：`build_rtc_inpainting_batch`、`rtc_velocity_loss`
  - Qwen3 流式适配：`Qwen3VLStreamRunnerSnapshot`、`export_selected_kv_cache`

## 子模块职责与接口契约

- `qwen3_stream/`
  - `Qwen3VLStreamRunnerSnapshot`：冻结原 stream runner 行为，防止上游接口漂移。
  - `export_selected_kv_cache`：从 `past_key_values` 提取指定层 KV，供动作专家注入。

- `action_expert/`
  - `ActionExpertBackbone.forward`：输入 `(noisy_action, state, time, kv_cache)`，输出速度场 `u_t`。
  - `euler_sample_actions`：欧拉积分采样动作 chunk，支持读写 `DiffusionKVCache`。
  - `ActionExpertRunner.sample`：面向业务调用的一站式采样接口。

- `pipeline/`
  - `validate_rtc_params`：校验调度参数合法性。
  - `stitch_action_for_execution`：按 RTC 规则拼接历史前缀与新 chunk 可执行段。
  - `roll_chunk_after_execution`：执行后滚动下一步缓存。
  - `RTCChunkScheduler.schedule`：输出 `(execute_chunk, next_prev_chunk)`。

- `training/`
  - `build_rtc_inpainting_batch`：构建训练时 RTC 样本（前缀 teacher-forced + 后缀去噪学习）。
  - `rtc_velocity_loss`：仅在 `loss_mask` 有效位置聚合 MSE。

## 关键“引用与呼应”关系

- `qwen3_stream/kv_export.py` → `action_expert/runner.py`
  - 前者输出 layer-wise KV 列表，后者在 `sample()` 传入扩散头。
- `training/loss_rtc.py` ↔ `pipeline/scheduler.py`
  - 训练侧通过 delay mask 学习“前缀已知、后缀预测”，推理侧用相同语义执行拼接与滚动。
## 类型约定

- 动作 chunk：`torch.Tensor`，形状 `[B, H, D]`
- 状态 token / state 条件：`torch.Tensor`，常见形状 `[B, Ds]` 或 `[B, H, Ds]`
- KV cache：`list[tuple[torch.Tensor, torch.Tensor]]`，每层 `(K, V)`
- token 标签：`torch.LongTensor`，形状 `[B, L]`

## 快速接入路径

1. 用 `Qwen3VLStreamRunnerSnapshot` 维护流式上下文。
2. 用 `export_selected_kv_cache()` 导出选层 KV。
3. 调用 `ActionExpertRunner.sample(state, kv_cache=...)` 生成动作 chunk。
4. 调用 `RTCChunkScheduler.schedule()` 产出可执行片段并滚动缓存。
5. 训练时替换为 `build_rtc_inpainting_batch + rtc_velocity_loss`。

## 配置解耦说明

- RTC 全链路默认使用 `configs/vla_qwen3_rtc.yaml`。
- 该配置不包含 `oat_tokenizer_checkpoint` 等旧 token/OAT 字段。
- 旧 `configs/vla_qwen3.yaml` 仅保留给历史 token 路径脚本。
