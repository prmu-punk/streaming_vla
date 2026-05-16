## 1) 核心 Idea（优先理解）
- **动作端维护一个 persistent action window；新 `obs` 到来时刷新视觉条件；动作窗口继续在最新条件下推进去噪，并尽快把最近待执行动作更新出来。**
- 这里的 action 端不是“一次生成一个完整 chunk 然后整体丢掉重来”，而是维护一个**persistent action window**。窗口前部尽量复用已经去噪过的 latent，窗口尾部只为新增未来步重新采样噪声。
- 这条路线的目标是提高 **reaction speed**：不频繁从头启动 action diffusion job，同时让最新观测尽快影响窗口前部的近端动作。
- 从方法定义上，我们希望这套策略尽量**不被外部 obs refresh 节奏绑定**。也就是说，观测何时到达、系统如何调度 refresh，不应成为方法本体的一部分；它们最多只影响实现效率，不应主导行为语义。

## 2) 当前方法定义（先按这个理解代码）
### 2.1 推理语义
- 新 `obs` 到来后，会生成该观测对应的**完整视觉条件**，当前 action window 后续的去噪步骤都应消费这份“最新条件”。
- action window 不是离散 chunk job，而是一个持续存在的 latent 窗口。它会随着环境步推进而 `shift`，并把“更近未来”的 token 尽快推向完成态。
- 当窗口左移时：
  - 旧窗口后半段 latent 会搬到新窗口前半段；
  - 对应 token 的去噪进度也会一起平移，表示这些动作继承已有计算；
  - 新出现的尾部 token 重新采样全新高斯噪声，并从头开始去噪。
- 然后在最新视觉条件下继续对整个窗口做一步去噪；任何在这一轮达到最终步数的 token，都应立即释放为可执行动作。

### 2.2 当前工程实现应如何看待
- 现有代码里仍然保留了双进程 `video worker` / `action worker` 的 async runtime，但这只是**当前工程路径**，不是方法本体。
- 当前 async runtime 的行为会受到 CUDA IPC、GPU 状态、EGL 渲染、queue 积压等系统因素强烈影响，因此它更适合被视为：
  - 一种 profiling / rollout scaffold；
  - 一种历史实现；
  - 一种待被更稳定执行方案替换的外部调度层。
- 阅读代码时要把“方法语义”和“当前 runtime 形态”分开：前者是 stable 的研究对象，后者只是暂时的执行载体。

### 2.3 训练范式
- 训练目标其实仅是让fastwam的ckpt能够适应每个token有不同的denoise step
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
- 也就是说，训练时一次只选一种完整观测条件来监督 action expert，而不是把 runtime 的外部 refresh 调度原样塞回训练。
- 当前默认配置是 `cache_mode=full_cur`，即用当前观测的完整 cache 去预测 action chunk。

### 2.4 “窗口尾部更新”在训练中的对应物
- 训练里通过 `token_noise_pattern` 来模拟窗口内“前后 token 状态不同”。
- 当前支持两种模式：
  - `random_all`
  - `front_low_high`
- 默认的 `front_low_high` 会对每个 token 独立采样扩散时刻 `t`，然后在 action 维度上按从小到大排序。
- 这意味着：
  - 窗口前部 token 更倾向于较小噪声 / 更接近完成态；
  - 窗口尾部 token 更倾向于较大噪声 / 更像“刚补进来的新噪声”。

## 3) 当前稳定结论（用于决策）
- 现在这条线的稳定方法叙事是：**obs refresh + persistent window shift **。
- 训练侧真正落地的是较保守的 action FT 方案：**冻结 video expert，仅训练 action expert 与 proprio encoder**。
- 当前多进程 async runtime 的 profile / rollout 结果，不能被直接当成“纯算法结论”；它们经常混入系统层不稳定性，尤其是 GPU transport、渲染与生命周期问题。
- 因此，分析时必须区分：
  - 方法本身是否更快更新近端动作；
  - 当前 runtime 是否把这种潜力稳定兑现出来。
- 评测上仍然建议优先使用 blocking 语义：环境在 formal phase 如果当前步动作还没准备好，就等待最新结果，而不是继续用 dummy action 硬推环境。
- 如果要判断这条路线是否真的成功，不能只看成功率，还要单独量化“新观测进入后，窗口前部动作被更新得有多快”。

## 4) 当前工作重点（下一阶段）
- 核心目标是验证这套方案是否真的提高了**反应速度**
- 训练侧如果要更贴近 runtime，可继续探索：
  - 更贴近 shift 后窗口状态的 token 噪声设计；
  - 是否需要把真实 runtime schedule replay 引入训练或蒸馏。
- 还需要把这套评估和训练逻辑迁移到 `robotwin` 更复杂任务上验证泛化。

## 5) 评测规范（必须遵守）
- 无论采用双进程 async 实现、单进程 unified 实现，比较实验都必须固定：
  - 控制 `dt`
  - obs stride / refresh 频率
  - `num_inference_steps`
  - `action_horizon`
  - GPU 资源预算
- 对比实验必须固定相同的 control dt 和相同的 obs stride，否则无法判断收益来自算法还是来自时序预算变化。
- 如果比较 blocking / non-blocking、warmup 与否、是否强制 first job，也必须单独记录，否则前缀动作丢弃量会改变结果解释。

## 6) 最小入口（只读这些关键文件）
- 推理核心：`src/fastwam/models/wan22/fastwam_streaming.py`
- 训练核心：`src/fastwam/models/wan22/streaming_backbone.py`
- cache / job 数据结构：`src/fastwam/models/wan22/streaming_cache.py`
- 当前 async runtime 实现：`src/fastwam/utils/async_streaming_runtime.py`
- 当前 worker 主循环：`src/fastwam/utils/async_streaming_workers.py`
- LIBERO 流式训练数据：`src/fastwam/datasets/lerobot/streaming_episode_dataset.py`
- 训练配置：`configs/task/libero_streaming_action_ft_2cam224_1e-4.yaml`
- 模型配置：`configs/model/fastwam_streaming.yaml`
- Blocking + profiled rollout：`experiments/libero/eval_libero_rollout_blocking_profiled.py`
- 单任务评测入口：`experiments/libero/eval_libero_single_profiled.py`
- layer source 统计聚合：`experiments/libero/collect_libero_layer_distribution.py`
- schedule replay / x_t 导出：`experiments/libero/build_xt_replay.py`

在阅读项目时，优先从上面这些入口进入。除非讨论明确涉及实验产物、日志或第三方实现，否则不要扩展去读结果 json、历史输出目录或无关细节文件。
