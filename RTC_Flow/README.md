# RTC_Flow

`RTC_Flow` 是 `Streaming_VLA` 下的 RTC 全链路独立工程入口，包含：
- 独立配置：`configs/`
- 运行脚本：`scripts/`
- 输出目录：`outputs/`

该目录默认走 **VLM KV 条件 + 扩散动作头 + RTC 异步调度**，不依赖旧 OAT token 训练接口。

## 目录结构

```text
RTC_Flow/
  configs/
    train_libero90_async.yaml
    vla_qwen3_rtc.yaml
    rtc_async_vla.yaml
  scripts/
    train_async.py
    eval_online.py
    rtc_rollout_utils.py
  outputs/
    runs/
    eval/
```

## 关键配置说明

- `configs/train_libero90_async.yaml`
  - `model.vla_config_path`: 指向 `configs/vla_qwen3_rtc.yaml`
  - `rtc_async.config_path`: 指向 `configs/rtc_async_vla.yaml`
  - `dataset.zarr_path`: 需要改成你的本地 zarr 数据路径
- `context_budget_tokens`: 仅用于离线上下文预算估算，不是 token 训练长度。

## 训练

在 `Streaming_VLA` 根目录执行：

```bash
python RTC_Flow/scripts/train_async.py --run-name rtc_flow_exp \
  --extra dataset.zarr_path=/abs/path/to/libero90.zarr training.num_epochs=10
```

输出默认写入：
- `RTC_Flow/outputs/runs/<run-name>/`

## 在线评估

```bash
python RTC_Flow/scripts/eval_online.py \
  --checkpoint /abs/path/to/checkpoints/best.pt \
  --task libero_spatial_task_name \
  --config RTC_Flow/configs/train_libero90_async.yaml \
  --inference-delay 2 --execute-horizon 4 --save-video

也可直接运行子项目主入口：

```bash
python RTC_Flow/scripts/train_libero90_async.py --config-path RTC_Flow/configs --config-name train_libero90_async
python RTC_Flow/scripts/eval_libero90_rtc_online.py --checkpoint /abs/path/to/best.pt --task <task_name> --config RTC_Flow/configs/train_libero90_async.yaml
```
```

## 依赖

请确保环境具备：
- `hydra-core`
- `omegaconf`
- `wandb`
- `imageio`
- `zarr`

并使用当前项目的 Python 环境运行。
