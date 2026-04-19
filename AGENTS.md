## 1) 核心 Idea（优先理解）
- 系统采用异步双专家协作：`video expert` 持续刷新 cache，`action expert` 以独立去噪时钟生成动作。
- 不再把多层 latent 作为“整包 packet”排队传递；改为一组可被并发访问和增量更新的 cache（概念上是固定 cache bank）。
- 在 action 去噪的任一步中，cache 各层来源可能混合于不同观测时刻（例如部分来自 `o_t`，部分来自 `o_{t-1}`）。
- 我们希望通过该idea达到的目标为：让一个chunk中的动作能够感知最新的obs，从而提升反应速度。
- 关键矛盾：在有限时序预算（job wall time vs control dt）下，兼顾成功率、稳定性与实时交付。

## 2) 已完成实现（可直接当工具使用）
- 异步推理运行时：已落地主进程 + video worker + action worker 的多进程结构。
- Layer 来源统计：已支持不截断 offset 的来源统计，可观察长尾来源。
- 分布聚合工具：可跨 task/episode 聚合 layer source，并导出训练可用分布建议。
- Blocking 评测工具：支持“控制前缀丢弃”并联动 profile 观察行为变化。
- 并行评测调度工具：支持多任务并发、GPU 负载管理与结果汇总。
- 时序对齐机制：rollout 支持最小步长对齐（min-step-dt）以减少统计语义偏差。

## 3) 当前稳定结论（用于决策）
- FT 模型有效，但能力更偏向异步/混合 cache 鲁棒性，当前异步推理成功率随微调时间上升而上升，初步认为有效。但不等价于全场景提升，也不一定意味着反应能力提升。
- 已观察到潜在遗忘：非混合cache的同步推理可能退化
- 时序预算上，目前可在dt=50ms下保持稳定，详情见配置文件与测试用命令：
python experiments/libero/eval_libero_single_profiled.py --config-name sim_libero task=libero_streaming_action_ft_2cam224_1e-4 model._target_=fastwam.runtime.create_fastwam_streaming ckpt=/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/FastWAM/runs/libero_streaming_action_ft_2cam224_1e-4/2026-04-09_15-
13-56/checkpoints/weights/step_002200.pt EVALUATION.num_trials=1 EVALUATION.task_suite_name=libero_10 EVALUATION.task_id=6 EVALUATION.async_video_device=cuda:0 EVALUATION.async_action_device=cuda:1 EVALUATION.async_obs_stride_env_steps=3 EVALUATION.async_action_trigger_every_n_obs=3 EVALUATION.num_inference_steps=8 EVALUATION.async_warmup_action_jobs=20 EVALUATION.async_control_dt_ms=50
- 推理侧去掉 dummy action 推进环境,也就是在环境测未收到动作时阻塞环境，防止前缀丢弃过多。
## 4) 当前工作重点（下一阶段）
- 理想目标是获得与原同步推理方式差不多甚至更高的成功率，同时想办法量化模型反应速度，要比原模型有提升（如加入环境扰动后能即使做出反应）
- 如何在robotwin的更复杂任务上测试
## 5) 评测规范（必须遵守）
- 异步 policy 对算力资源敏感，评测必须固定资源切分。
- 推荐隔离：2 张 GPU 专供 policy（video/action worker）
- 对比实验必须固定相同资源配置与 dt；否则结果不具可比性。

## 6) 最小入口（仅保留关键）
- 并行评测入口：`experiments/libero/run_libero_parallel_test.sh`
- 全量/串行 profile：`experiments/libero/eval_libero_all_serial_profiled.py`
- Blocking + profiled：`experiments/libero/eval_libero_rollout_blocking_profiled.py`
- 分布聚合：`experiments/libero/collect_libero_layer_distribution.py`
- 训练核心实现：`src/fastwam/models/wan22/fastwam_streaming.py`

在阅读项目时，你只需要从上面的入口文件进入，阅读部分较重要的代码文件即可。json等实验结果文件以及其他更加细节的代码，除非我们的讨论中涉及到这些内容，否则不要阅读

