# FastWAM 异步训练简版思路

## 目标

1. 推理延迟目标：warmup 后 `action_job` 稳定在串行同级（约 380ms）。
2. 质量目标：异步 cache 混合条件下，动作质量和任务成功率不下降。

## 核心问题

当前训练的 cache 混合规则是手工设定的，和真实异步 rollout 中每个 denoise step 的 cache 状态不一定一致，导致分布错配。

## 总体方案（三步）

1. **先观测真实分布**
   - 在异步 rollout 中记录每个 denoise step 的 cache 状态统计（如 frontier、layer age）。
   - 先不改模型，只做数据采样统计。

2. **再做分布对齐训练**
   - 训练时按真实统计分布采样 mixed cache。
   - 替代当前固定 bucket/固定规则。

3. **最后做轻量鲁棒增强**
   - 在真实分布附近加小扰动（少量更旧/更极端 frontier）。
   - 提高调度抖动下稳定性。


## 训练改动建议（最小版）

1. 保留现有 streaming loss 主体。
2. 将 mixed cache 采样改为“读取真实分布配置”。
3. 可选增加一个小权重一致性项（mixed 输出接近 fresh 输出）来减小异步漂移。

## 最小实验集

1. E1：现有规则（baseline）
2. E2：真实分布采样
3. E3：真实分布采样 + 小权重一致性项

对比：
- 成功率
- `action_job` / `action_job_wall`
- `actions_missed` / `dropped_prefix_actions`

## 结论

先采集真实异步分布，再按分布训练，比继续手工调规则更稳妥，也更容易解释结果。
