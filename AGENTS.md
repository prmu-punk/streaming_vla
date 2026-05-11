## 1) 核心 Idea（优先理解）
- 当前版本的流式推理主线是：**新 `obs` 到来 -> 刷新当前视觉条件 -> action 窗口按最新 `env_step` 做 `shift` -> 窗口尾部补全全新噪声 token -> 持续去噪并按 token 完成时释放动作**。
- 这里的 action 端不是“一次生成一个完整 chunk 然后整体丢掉重来”，而是维护一个**persistent action window**。窗口前部尽量复用已经去噪过的 latent，窗口尾部只为新增未来步重新采样噪声。
- 当前代码里的 video 侧已经不是“逐层增量 cache frontier 推进”训练范式；推理时每次新观测都会构造一份**完整 video cache**，action worker 始终消费自己手头最新的一份完整 snapshot。
- 这套机制的目标是：在不频繁从头启动 action diffusion job 的前提下，让动作窗口尽快对最新观测做出条件更新，同时减少前缀动作浪费。
- 核心矛盾仍然是：在有限 wall time / control dt / GPU 资源下，同时兼顾成功率、稳定性和反应速度。

## 2) 当前实现（先按这个理解代码）
### 2.1 推理运行时
- 运行时仍然是双进程：主进程 + `video worker` + `action worker`。
- `video worker` 收到新观测后，会直接构造该观测对应的**完整 video KV cache**，并把整份 cache 发送给 `action worker`。
- `action worker` 本地只保留“最新完整 cache”；如果短时间内来了多次更新，会丢弃中间旧更新，只消费最新一份。
- `action worker` 内部维护一个 `persistent=True` 的 `StreamingActionJob`。它不会自然结束，而是随着新观测持续推进。

### 2.2 窗口 shift 语义
- 每个 action token 都绑定一个 `token_env_steps`，表示它当前对应哪个环境步。
- 当最新 snapshot 的 `env_step` 大于当前 `window_start_env_step` 时，窗口会左移 `shift_steps = latest_env_step - window_start_env_step`。
- 左移后：
  - 旧窗口后半段 latent 会搬到新窗口前半段，表示“这些未来动作现在变成了更近的未来”。
  - 对应 token 的去噪计数 `token_denoise_counts` 也会一起平移，所以前部 token 会继承已有去噪进度。
  - 新出现的尾部 token 会重新采样**全新高斯噪声**，其去噪计数清零。
- 然后 action worker 用最新 cache 对整个窗口再做一步去噪；任何在这一轮达到最终步数的 token，会立刻通过 `just_released_mask` 释放为可执行动作。

### 2.3 当前训练范式
- 当前训练不是直接复现 runtime 里的“混合层来源异步 cache”分布，也不是训练一个显式的窗口 shift 模块。
- 训练数据采用 **episode-style 四帧样本**：
  - `obs_prev`
  - `obs_cur`
  - `obs_next`
  - `obs_next2`
  - 再配合从 `trigger_obs_idx` 开始的 `target_action` chunk 与 `proprio_t`
- 当前 `streaming_train.cache_mode` 只支持单一完整 cache 条件：
  - `full_prev`
  - `full_cur`
  - `full_next`
  - `full_next2`
- 也就是说，训练时一次只选一种完整观测条件来监督 action expert，而不是在层级上混合多时刻 cache。
- 当前默认配置是 `cache_mode=full_cur`，即用当前观测的完整 cache 去预测 action chunk。

### 2.4 当前“窗口尾部更新”在训练中的对应物
- 训练里通过 `token_noise_pattern` 来模拟窗口内“前后 token 状态不同”。
- 当前支持两种模式：
  - `random_all`
  - `front_low_high`
- 默认的 `front_low_high` 会对每个 token 独立采样扩散时刻 `t`，然后在 action 维度上按从小到大排序。
- 这意味着：
  - 窗口前部 token 更倾向于较小噪声 / 更接近完成态；
  - 窗口尾部 token 更倾向于较大噪声 / 更像“刚补进来的新噪声”。
- 因此，训练并不是显式执行一次 runtime shift，而是用 **“单 cache 条件 + token 级噪声前低后高”** 去逼近 persistent window 在在线推理中的状态分布。

## 3) 当前稳定结论（用于决策）
- 现在代码主线已经明确转向 **obs refresh + persistent window shift + tail re-noise**，而不是旧文档里那种“多层 cache bank 异步混合训练”叙事。
- 训练侧真正落地的是较保守的 action FT 方案：**冻结 video expert，仅训练 action expert 与 proprio encoder**。
- 现有 profile / 分布统计工具仍然有价值，但它们现在更适合做**推理行为分析**和**训练假设设计**；默认训练路径本身并不会直接读取这些分布去做 mixed-cache supervision。
- 评测上仍然建议使用 blocking 语义：环境在 formal phase 如果当前步动作还没准备好，就等待最新结果，而不是继续用 dummy action 硬推环境。
- 如果要判断这条路线是否真的成功，不能只看成功率，还要单独量化“新观测进入后，窗口前部动作被更新得有多快”。

## 4) 当前工作重点（下一阶段）
- 核心目标是验证这套新方案是否真的提高了**反应速度**，而不只是让模型对不同观测时刻更鲁棒。
- 需要把“成功率”和“响应性”拆开评测。理想指标包括：
  - 最新观测进入后，多久能影响到最近几个待执行 action token；
  - 在扰动注入后，窗口前部动作是否更快偏离旧计划；
  - 在同样 dt 和资源下，成功率是否接近或超过同步基线。
- 训练侧接下来如果要更贴近 runtime，可以继续探索：
  - 不同 `cache_mode` 的混合采样；
  - 更贴近 shift 后窗口状态的 token 噪声设计；
  - 是否需要把真实 runtime schedule replay 引入训练或蒸馏。
- 还需要把这套评估和训练逻辑迁移到 `robotwin` 更复杂任务上验证泛化。

## 5) 评测规范（必须遵守）
- 流式 policy 对资源切分非常敏感，评测时必须固定：
  - `async_video_device`
  - `async_action_device`
  - `async_obs_stride_env_steps`
  - `async_control_dt_ms`
  - `num_inference_steps`
- 推荐使用 2 张 GPU 隔离 video / action worker；如果只用 1 张卡，对比实验必须保证所有方法资源完全一致。
- 对比实验必须固定相同的 control dt 和相同的 obs stride，否则无法判断收益来自算法还是来自时序预算变化。
- 如果比较 blocking / non-blocking、warmup 与否、是否强制 first job，也必须单独记录，否则前缀动作丢弃量会改变结果解释。

## 6) 最小入口（只读这些关键文件）
- 推理核心：`src/fastwam/models/wan22/fastwam_streaming.py`
- 训练核心：`src/fastwam/models/wan22/streaming_backbone.py`
- cache / job 数据结构：`src/fastwam/models/wan22/streaming_cache.py`
- 异步 runtime：`src/fastwam/utils/async_streaming_runtime.py`
- worker 主循环：`src/fastwam/utils/async_streaming_workers.py`
- LIBERO 流式训练数据：`src/fastwam/datasets/lerobot/streaming_episode_dataset.py`
- 训练配置：`configs/task/libero_streaming_action_ft_2cam224_1e-4.yaml`
- 模型配置：`configs/model/fastwam_streaming.yaml`
- Blocking + profiled rollout：`experiments/libero/eval_libero_rollout_blocking_profiled.py`
- 单任务评测入口：`experiments/libero/eval_libero_single_profiled.py`
- layer source 统计聚合：`experiments/libero/collect_libero_layer_distribution.py`
- schedule replay / x_t 导出：`experiments/libero/build_xt_replay.py`

在阅读项目时，优先从上面这些入口进入。除非讨论明确涉及实验产物、日志或第三方实现，否则不要扩展去读结果 json、历史输出目录或无关细节文件。

