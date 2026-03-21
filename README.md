# Streaming_VLA RTC Flow

这是当前 `Streaming_VLA` 仓库的 **离线训练 + 在线推理** RTC 异步控制版本。  
核心思想是将动作生成与 token 解码路径解耦，采用：

- VLM 流式上下文编码（Qwen3）
- 从 VLM 导出 KV-Cache 作为条件
- 扩散/流匹配动作专家生成动作 chunk
- RTC 调度器按 `delay/horizon` 异步拼接并执行动作

---

## 宏观架构

整体执行链路如下：

1. 视觉帧、状态、任务语言输入到 VLM 编码器；
2. 从 VLM 的 `past_key_values` 中导出指定层 KV 条件；
3. `ActionExpertRunner` 在 KV 条件下预测动作 chunk；
4. `RTCChunkScheduler` 根据 `inference_delay` 与 `execute_horizon` 形成可执行片段；
5. 在线循环中执行 `execute_chunk`，再滚动进入下一控制周期。

对应模块分层：

- `dataset/`：libero90 数据读取与离线 context 采样
- `model/vla_qwen3_rtc.py`：离线批量上下文编码与 KV 输出
- `model/vla_qwen3_rtc_online.py`：在线统一入口 `Qwen3RTCVLAOnlinePipeline`
- `model/rtc_async/`：动作专家、RTC 训练损失、调度器、stream 适配
- `scripts/`：训练与评估入口脚本
- `configs/`：训练/VLM/RTC 三类配置

---

## 文件树

```text
Streaming_VLA/
├── .python-version                     # Python 版本声明（uv 使用）
├── pyproject.toml                      # 项目依赖定义
├── uv.lock                             # 依赖锁文件
├── README.md
├── configs/
│   ├── train_libero90_async.yaml       # 训练总配置（数据/优化器/入口路径）
│   ├── vla_qwen3_rtc.yaml              # VLM 编码器配置（模型路径、stream gate）
│   └── rtc_async_vla.yaml              # RTC 与动作专家配置（delay/horizon 等）
├── dataset/
│   ├── __init__.py
│   ├── libero90_async_dataset.py       # 基础 episode 数据读取
│   ├── libero90_async_offline_context_dataset.py  # 训练样本构造（context/anchor/target）
│   └── bucket_sampler.py               # 与主干对齐的变长 bucket 采样
├── model/
│   ├── __init__.py
│   ├── template_qwen3_vla.py
│   ├── vla_qwen3_rtc.py                # 训练侧 VLM 编码器（离线上下文 -> KV）
│   ├── vla_qwen3_rtc_online.py         # 在线推理管线入口
│   ├── qwen3_vl/                       # 本地化 Qwen3-VL 组件（配置/处理/stream runner）
│   └── rtc_async/
│       ├── __init__.py
│       ├── README.md
│       ├── action_expert/              # 动作专家网络、采样与 runner
│       ├── pipeline/                   # RTC 调度逻辑
│       ├── qwen3_stream/               # KV 导出与 stream snapshot
│       └── training/                   # RTC 训练批构造与损失
├── scripts/
│   ├── __init__.py
│   ├── train_async.py                  # 训练包装入口
│   ├── train_libero90_async.py         # 训练薄包装
│   ├── eval_online.py                  # 评估包装入口
│   ├── eval_libero90_rtc_online.py     # 在线评估薄包装
│   └── smoke_test_full.py              # 端到端冒烟测试
├── workspace/
│   ├── train_libero90_async.py         # 训练主实现
│   └── eval_libero90_rtc_online.py     # 在线评估主实现
```

---

## 核心接口

### 1) 训练侧接口

- `Qwen3RTCVLAEncoder.forward_offline_context_batch(...)`
  - 输入：离线样本列表（`context_* / anchor_* / target_chunk`）
  - 输出：`target_chunk + past_key_values + attention_mask`
- `export_selected_kv_cache(...)`
  - 将 VLM 全层 KV 裁剪为 `selected_layers` 对应层
- `build_rtc_inpainting_batch(...)`
  - 构造训练期延迟掩码与噪声输入
- `rtc_velocity_loss(...)`
  - 在有效后缀位置聚合速度场损失
- `ActionExpertRunner.forward(...)`
  - 输入 `noisy_action/state/time/kv_cache`，输出 `pred_u_t`

### 2) 推理侧接口

- `Qwen3RTCVLAOnlinePipeline.reset(prompt)`
  - 重置在线上下文与调度状态
- `Qwen3RTCVLAOnlinePipeline.push_observation(frames, state, ts_ms, num_frames)`
  - 写入当前观测到流式上下文
- `Qwen3RTCVLAOnlinePipeline.sample_and_schedule(...)`
  - 采样动作 chunk 并输出 `execute_chunk`
- `Qwen3RTCVLAOnlinePipeline.set_runtime_schedule_params(...)`
  - 运行时覆盖 `inference_delay / execute_horizon`

---

## 配置关系

训练主配置：`configs/train_libero90_async.yaml`

- `model.vla_config_path` -> `configs/vla_qwen3_rtc.yaml`
- `rtc_async.config_path` -> `configs/rtc_async_vla.yaml`
- `dataset.zarr_path`：默认已指向本地 `libero10_N500` zarr 路径；可按需覆盖

VLM 配置：`configs/vla_qwen3_rtc.yaml`

- `model_name_or_path`：本地模型路径或 HF 模型名
- `stream.state_interval_s / vision_interval_s`：在线流式写入门控间隔

RTC 配置：`configs/rtc_async_vla.yaml`

- `rtc.inference_delay`：推理延迟 `d`
- `rtc.execute_horizon`：执行窗口 `h`
- `rtc.simulated_delay`：训练期随机延迟上限
- `action_expert.*`：动作专家网络结构与采样步数

---

## 训练脚本（CUDA）

推荐在本项目目录执行，注意对应脚本和数据集路径的修改：

```bash
uv sync --frozen
```

### 方式 1：包装入口（推荐）

```bash
cd /home/luye/data/Streaming_VLA
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run python scripts/train_async.py \
  --run-name rtc_flow_exp_v1 \
  --extra \
    dataset.zarr_path=/home/luye/data/Streaming_VLA/data/libero/libero10_N500.zarr/libero10_N500.zarr \
    training.num_epochs=10 \
    dataloader.batch_size=4
```

### 方式 2：直接调用主训练脚本

```bash
cd /home/luye/data/Streaming_VLA
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run python scripts/train_libero90_async.py \
  --config-path /home/luye/data/Streaming_VLA/configs \
  --config-name train_libero90_async \
  dataset.zarr_path=/home/luye/data/Streaming_VLA/data/libero/libero10_N500.zarr/libero10_N500.zarr \
  hydra.run.dir=/home/luye/data/Streaming_VLA/outputs/runs/manual_run
```

说明：

- `CUDA_VISIBLE_DEVICES=0` 指定使用第 0 块 GPU；多卡时可改为 `0,1` 等。
- `hydra.run.dir` 必须是本机可写的真实绝对路径，不要使用 `/abs/path/...` 这类占位符。

训练输出常见位置：

- `/home/luye/data/Streaming_VLA/outputs/runs/<run-name>/`
- checkpoint：`.../checkpoints/best.pt`

---

## 在线测试样例

包装入口：

```bash
uv run python scripts/eval_online.py \
  --checkpoint /abs/path/to/checkpoints/best.pt \
  --task libero_spatial_task_name \
  --config configs/train_libero90_async.yaml \
  --match-rank 0 \
  --num-frames 6 \
  --max-control-cycles 120 \
  --inference-delay 2 \
  --execute-horizon 4 \
  --save-video
```

直接主脚本：

```bash
uv run python scripts/eval_libero90_rtc_online.py \
  --checkpoint /abs/path/to/checkpoints/best.pt \
  --config configs/train_libero90_async.yaml \
  --task libero_spatial_task_name \
  --match-rank 0 \
  --num-frames 6 \
  --max-control-cycles 120 \
  --inference-delay 2 \
  --execute-horizon 4 \
  --save-video
```

评估脚本会输出 JSON 指标（success、total_env_steps、配置回显等），启用 `--save-video` 时会写入视频文件。

---

## 依赖与运行环境

请使用 `uv` 管理环境与依赖（项目已提供 `pyproject.toml`、`uv.lock`、`.python-version`）：

```bash
cd /home/luye/data/Streaming_VLA
uv sync --frozen
```

若你在 `Streaming_VLA` 根目录执行脚本，直接使用 `uv run` 即可自动使用该环境。

默认关键依赖由锁文件统一管理（例如 `torch`、`hydra-core`、`omegaconf`、`wandb`、`imageio`、`zarr`、`numpy`）。
