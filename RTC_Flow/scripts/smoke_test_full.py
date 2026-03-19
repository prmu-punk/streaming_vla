#!/usr/bin/env python3
"""完整的 RTC_Flow 冒烟测试脚本 (无需真实数据集)

此脚本将:
1. 加载本地下载的 RynnBrain-2B 权重作为 VLM 编码器。
2. 实例化 ActionExpert 动作专家。
3. 伪造一些输入图像、状态和文本提示。
4. 执行一次完整的离线前向传播 (VLM 编码 -> KV 导出 -> RTC Inpainting 动作去噪)。
"""

from __future__ import annotations

import gc
import os
import sys
import torch
import numpy as np

# 确保能找到内部模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder, RTCVLAConfig, StreamConfig, OfflineContextSample
from model.rtc_async.action_expert.runner import ActionExpertRunner, ActionExpertRunnerConfig
from model.rtc_async.training.loss_rtc import build_rtc_inpainting_batch, rtc_velocity_loss


def print_mem(tag: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"[Memory] {tag} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def main():
    print("=== 开始 RTC_Flow 完整冒烟测试 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用的设备: {device}")

    # 1. 准备配置
    model_path = "/data/luye/Streaming_VLA/RTC_Flow/models/RynnBrain-2B"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    vla_cfg = RTCVLAConfig(
        model_name_or_path=model_path,
        state_dim=16,
        device=device,
        stream=StreamConfig()
    )

    ae_cfg = ActionExpertRunnerConfig(
        state_dim=16,
        action_dim=7,
        horizon=10,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_inference_steps=5
    )

    # 2. 初始化模型
    print("\n>>> 正在加载 Qwen3RTCVLAEncoder (这可能需要 1-2 分钟和 ~4.5GB 显存) ...")
    # Qwen3RTCVLAEncoder 期望接收 yaml 文件的路径
    vla_config_path = "/home/luye/data/Streaming_VLA/RTC_Flow/configs/vla_qwen3_rtc.yaml"
    vla_encoder = Qwen3RTCVLAEncoder(vla_config_path).to(device)
    # 设为 eval 模式，节省显存
    vla_encoder.eval()
    print_mem("VLA Loaded")

    print("\n>>> 正在初始化 ActionExpertRunner ...")
    action_expert = ActionExpertRunner(ae_cfg).to(device)
    print_mem("Expert Loaded")

    # 3. 构造伪造数据 (Batch Size = 2)
    print("\n>>> 构造离线伪造数据 ...")
    B = 2
    samples = []
    for _ in range(B):
        # 伪造一个 2 帧的视频序列 (H, W)=(224, 224), 3 通道
        # Qwen3-VL 期望视频输入形状为 (T, H, W, C)
        frames = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8)
        
        # 伪造两个历史状态
        states = torch.randn(2, 16)
        
        sample: OfflineContextSample = {
            "instruction": "Pick up the red block.",
            "context_videos": frames,
            "context_states": states,
            "context_time_steps": [0.0, 0.1],
            "context_time_indices": torch.tensor([0, 2], dtype=torch.long), # 假装步数索引
            "anchor_time_idx": torch.tensor(2, dtype=torch.long),
            "anchor_video": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "anchor_state": torch.randn(16),
            "target_chunk": torch.randn(10, 7) # horizon=10, action_dim=7
        }
        samples.append(sample)

    # 伪造真实的 action 标签 (Batch Size, Horizon, Action Dim)
    true_action = torch.randn(B, ae_cfg.horizon, ae_cfg.action_dim, device=device)

    # 4. 执行前向传播
    print("\n>>> 执行 VLA 编码与 KV-Cache 提取 ...")
    with torch.no_grad(): # VLA 通常被冻结
        vla_out = vla_encoder.forward_offline_context_batch(samples=samples, num_frames=2)
    
    # 模拟 export_selected_kv_cache 的行为 (提取指定层)
    # transformers 默认返回的是 DynamicCache 对象，可以通过 .key_cache 取出 tuple 列表
    all_kvs = vla_out["past_key_values"]
    # transformers 默认返回的是 DynamicCache 对象
    if type(all_kvs).__name__ == "DynamicCache" or type(all_kvs).__name__ == "Cache":
        if hasattr(all_kvs, "key_cache"):
            all_kvs_list = list(zip(all_kvs.key_cache, all_kvs.value_cache))
        else:
            # 兼容其他类型的 cache 对象 (比如 qwen3_vl 里的自定义 Cache)
            # 很多自定义 cache 用 keys / values
            if hasattr(all_kvs, "keys") and hasattr(all_kvs, "values"):
                 # 如果 all_kvs 本身就是一个 Cache 对象而不是 list，我们需要将其包装为 tuple list
                 # 假设只有一个 cache 对象，那么 selected_kvs 只有一层。
                 # qwen3_vl_text 实际上返回的是 list[Cache]
                 pass
    
    # 强制将 qwen3_vl 返回的 list[Cache] 转换为 list[tuple(k, v)]
    if isinstance(all_kvs, list) and len(all_kvs) > 0 and hasattr(all_kvs[0], "keys") and hasattr(all_kvs[0], "values"):
        all_kvs_list = [(c.keys, c.values) for c in all_kvs]
    elif hasattr(all_kvs, "to_legacy_cache"):
        all_kvs_list = all_kvs.to_legacy_cache()
    elif type(all_kvs).__name__ == "DynamicCache":
         # DynamicCache 的键值一般通过迭代器或者 .key_cache 访问，某些版本中是列表 self.key_cache
         if hasattr(all_kvs, "key_cache"):
             all_kvs_list = list(zip(all_kvs.key_cache, all_kvs.value_cache))
         else:
             # 如果没有，可能本身可迭代
             all_kvs_list = list(all_kvs)
    elif isinstance(all_kvs, tuple) and isinstance(all_kvs[0], tuple):
        all_kvs_list = list(all_kvs)
    else:
        raise ValueError(f"未知的 KV Cache 格式: {type(all_kvs)}")
        
    selected_kvs = all_kvs_list[-3:] if len(all_kvs_list) >= 3 else all_kvs_list
    print(f"提取了 {len(selected_kvs)} 层 KV-Cache.")
    print_mem("After VLA Forward")

    print("\n>>> 构建 RTC Inpainting Batch 并模拟延迟 ...")
    rtc_batch = build_rtc_inpainting_batch(
        action=true_action,
        simulated_delay=5 # 假设最大延迟 5 步
    )

    print("\n>>> 执行 Action Expert 速度预测 ...")
    # Action Expert 期望的 KV 格式是 list[tuple[Tensor, Tensor]]
    # 确保 selected_kvs 的元素是 (k, v)
    formatted_kvs = []
    for layer_kv in selected_kvs:
        if hasattr(layer_kv, "keys") and hasattr(layer_kv, "values"):
            formatted_kvs.append((layer_kv.keys, layer_kv.values))
        elif isinstance(layer_kv, tuple) and len(layer_kv) == 2:
            formatted_kvs.append(layer_kv)
        elif isinstance(layer_kv, tuple) and len(layer_kv) == 3 and layer_kv[2] is None:
            # 魔改的 Qwen3 返回了 (key, value, None) 的三元组
            formatted_kvs.append((layer_kv[0], layer_kv[1]))
        elif isinstance(layer_kv, tuple) and hasattr(layer_kv[0], "keys"):
            # 有时会返回 tuple(Cache(),) 这种嵌套结构
            formatted_kvs.append((layer_kv[0].keys, layer_kv[0].values))
        else:
            raise ValueError(f"层 KV 格式不支持拆包: {type(layer_kv)}, 内容长度: {len(layer_kv) if isinstance(layer_kv, tuple) else 'N/A'}")

    pred_u_t = action_expert(
        noisy_action=rtc_batch.x_t,
        state=torch.stack([s["context_states"][-1] for s in samples]).to(device),
        time=rtc_batch.time,
        kv_cache=formatted_kvs,
        attention_mask=vla_out["attention_mask"]
    )
    print(f"预测的速度场形状: {pred_u_t.shape}")

    print("\n>>> 计算 RTC 损失 ...")
    loss = rtc_velocity_loss(pred_u_t=pred_u_t, batch=rtc_batch)
    print(f"RTC Velocity Loss: {loss.item():.4f}")

    print("\n=== 冒烟测试全部通过！🚀 ===")

if __name__ == "__main__":
    main()
