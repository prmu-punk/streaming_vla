# -*- coding: utf-8 -*-
"""
GPU 占用程序 - 保持 GPU 利用率高，CPU 占用低

用法:
    python gpu_occupy.py                    # 占用所有可见 GPU
    python gpu_occupy.py --gpus 0,1         # 只占用指定 GPU
    python gpu_occupy.py --util 80          # 目标利用率 80%
    python gpu_occupy.py --mem 0.5          # 占用 50% 显存

按 Ctrl+C 退出
"""

import argparse
import signal
import time
import torch


def _allocate_memory_buffers(device: torch.device, target_mem_bytes: int, chunk_bytes: int = 1 << 30):
    """
    Allocate uint8 buffers to occupy GPU memory precisely.
    Using byte buffers avoids the square-matrix size cap problem.
    """
    buffers = []
    allocated = 0
    while allocated < target_mem_bytes:
        cur = min(chunk_bytes, target_mem_bytes - allocated)
        buffers.append(torch.empty(cur, device=device, dtype=torch.uint8))
        allocated += cur
    return buffers


def occupy_gpu(
    device_ids: list = None,
    target_util: float = 90.0,
    mem_fraction: float = 0.8,
    matrix_size: int = 4096,
):
    """
    占用指定 GPU，保持高利用率

    Args:
        device_ids: GPU 设备 ID 列表，None 表示所有可见 GPU
        target_util: 目标 GPU 利用率 (0-100)
        mem_fraction: 显存占用比例 (0-1)
        matrix_size: 矩阵大小，越大 GPU 利用率越高
    """
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    # 确定要占用的 GPU
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if not device_ids:
        print("没有可用的 GPU")
        return

    print(f"准备占用 GPU: {device_ids}")
    print(f"目标利用率: {target_util}%")
    print(f"显存占用比例: {mem_fraction * 100}%")
    print(f"矩阵大小: {matrix_size}x{matrix_size}")
    print("按 Ctrl+C 退出\n")

    # 信号处理
    stop_flag = [False]

    def signal_handler(signum, frame):
        print("\n收到退出信号，正在清理...")
        stop_flag[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 在每个 GPU 上创建张量和计算
    tensors = {}

    for dev_id in device_ids:
        device = torch.device(f"cuda:{dev_id}")

        # 获取显存信息
        total_mem = torch.cuda.get_device_properties(dev_id).total_memory
        target_mem = int(total_mem * mem_fraction)
        # 计算张量额外也会临时申请显存，给工作区预留一点空间，避免 OOM。
        reserve_mem = max(2 << 30, int(total_mem * 0.05))
        buffer_target = max(0, target_mem - reserve_mem)
        size = int(matrix_size)

        print(f"GPU {dev_id}: {torch.cuda.get_device_name(dev_id)}")
        print(f"  总显存: {total_mem / 1024**3:.1f} GB")
        print(f"  目标占用: {target_mem / 1024**3:.1f} GB")
        print(f"  预留工作区: {reserve_mem / 1024**3:.1f} GB")
        print(f"  buffer占用目标: {buffer_target / 1024**3:.1f} GB")
        print(f"  计算矩阵尺寸: {size}x{size}")

        # 用 byte buffer 精确占显存
        mem_buffers = _allocate_memory_buffers(device, buffer_target)

        # 计算张量单独创建，用于维持 GPU 利用率
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)

        tensors[dev_id] = (device, mem_buffers, a, b)

        # 显示实际占用
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(dev_id)
        print(f"  实际占用: {allocated / 1024**3:.2f} GB\n")

    # 计算循环
    iteration = 0
    start_time = time.time()

    # 根据目标利用率调整休眠时间
    # 利用率越低，休眠越多
    sleep_time = max(0, (100 - target_util) / 1000)  # 简单的线性关系

    try:
        while not stop_flag[0]:
            for dev_id, (device, mem_buffers, a, b) in tensors.items():
                # 矩阵乘法 - GPU 密集型操作
                c = torch.mm(a, b)

                # 再做一些操作保持 GPU 忙碌
                c = torch.nn.functional.relu(c)
                c = torch.mm(c, a.t())

                # 确保计算完成
                torch.cuda.synchronize(device)

            iteration += 1

            # 每 100 次迭代打印状态
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {iteration}, 运行时间: {elapsed:.1f}s", end="\r")

            # 控制利用率的休眠
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n发生错误: {e}")

    finally:
        # 清理
        print("\n清理 GPU 内存...")
        tensors.clear()
        torch.cuda.empty_cache()
        print("已退出")


def main():
    parser = argparse.ArgumentParser(description="GPU 占用程序")
    parser.add_argument(
        "--gpus",
        type=str,
        default=1,
        help="要占用的 GPU ID，逗号分隔，如 '0,1,2'。默认占用所有可见 GPU"
    )
    parser.add_argument(
        "--util",
        type=float,
        default=60.0,
        help="目标 GPU 利用率 (0-100)，默认 90"
    )
    parser.add_argument(
        "--mem",
        type=float,
        default=0.5,
        help="显存占用比例 (0-1)，默认 0.8"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="最小矩阵大小，默认 4096"
    )
    args = parser.parse_args()

    if not (0 < args.mem < 1):
        raise ValueError(f"--mem 必须在 (0, 1) 之间，当前为 {args.mem}")
    if not (0 < args.util <= 100):
        raise ValueError(f"--util 必须在 (0, 100] 之间，当前为 {args.util}")

    # 解析 GPU ID
    device_ids = None
    if args.gpus:
        device_ids = [int(x.strip()) for x in args.gpus.split(",")]

    occupy_gpu(
        device_ids=device_ids,
        target_util=args.util,
        mem_fraction=args.mem,
        matrix_size=args.size,
    )


if __name__ == "__main__":
    main()
