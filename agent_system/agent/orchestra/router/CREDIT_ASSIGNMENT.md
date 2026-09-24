# Router Orchestra Credit Assignment

本文档记录 Router Orchestra 当前确定的 Router credit 和 SubAgent credit
设计。它描述 trainer 在 rollout 数据上应如何构造分组、计算 reward baseline，
以及把 credit 交给哪个 Agent。

## 1. 基本轨迹结构

每个 SubAgent segment 结束时，模型输出：

```text
<handoff>CONTINUE</handoff>
```

或：

```text
<handoff>HANDOFF</handoff>
```

`CONTINUE` 保持当前 SubAgent，不产生新的 Router 路由节点。
`HANDOFF` 结束当前 segment，并创建一个 Router 节点。Router 随后选择一个
SubAgent，或者选择 `FINISH` 提交当前结果。

例如：

```text
SubAgent 1
  HANDOFF
    Router 选择 SubAgent 2
      HANDOFF
        Router 选择 SubAgent 3
```

这里的 Agent 路径是：

```text
SubAgent 1 → SubAgent 2 → SubAgent 3
```

路径只在 Router 选择一个新的 SubAgent 时追加。SubAgent 的多次
`CONTINUE` response 仍属于当前路径下的同一个 segment。Router 在 HANDOFF 后
重新选择同一个 SubAgent 时，也要追加一次，例如：

```text
SubAgent 1 → SubAgent 2 → SubAgent 2
```

## 2. 轨迹标识和 prefix

相关字段的含义如下：

| 字段 | 含义 |
| --- | --- |
| `uid` | 同一个问题对应的一组 rollout，用于 Router 的 task-specific 分组 |
| `traj_uid` | 一条具体完整 trajectory 的唯一标识 |
| `agent_path` | 当前 trajectory 已经经过的 SubAgent 序列 |
| `prefix_key` | `tuple(agent_path)`，表示当前 SubAgent 所处的路径 prefix |
| `parent_prefix_key` | Router 选择下一个 SubAgent 前的 prefix |

例如：

```python
agent_path = ["SubAgent 1", "SubAgent 2"]
prefix_key = ("SubAgent 1", "SubAgent 2")
```

如果 Router 在这个节点选择 `SubAgent 3`，则 child prefix 为：

```python
child_prefix_key = (
    "SubAgent 1",
    "SubAgent 2",
    "SubAgent 3",
)
```

同一条 trajectory 中可能有多个 prefix。不同 trajectory 如果拥有相同的
prefix，可以进入同一个 SubAgent prefix group。

`traj_uid` 不用于定义 SubAgent baseline group，否则每条 trajectory 都会成为
一个只有单个样本的 group。它仍然需要用于去重：同一 trajectory 中同一个
prefix 可能有多个 response，但在计算 group 平均 reward 时只能统计一次。

## 3. SubAgent credit

SubAgent credit 是 prefix-conditioned 的 trajectory-level GRPO。

### 3.1 SubAgent 分组

SubAgent group 只使用 `prefix_key`：

```python
subagent_group_key = prefix_key
```

它不要求 `uid` 特异。也就是说，来自不同 `uid` 的 trajectory，只要具有相同
的 prefix，就可以共享同一个 SubAgent baseline。

计算平均 reward 前，先按照：

```text
(traj_uid, prefix_key)
```

去重，使一条 trajectory 在该 prefix group 中只出现一次。

### 3.2 Credit 公式

设 `G(p)` 是 prefix `p` 对应的 trajectory 集合，`R(τ)` 是 trajectory `τ`
的最终 reward：

\[
\mu(p)=\frac{1}{|G(p)|}\sum_{\tau\in G(p)}R(\tau)
\]

当前 trajectory 在 prefix `p` 下的 SubAgent credit 为：

\[
A_{\text{sub}}(\tau,p)=R(\tau)-\mu(p)
\]

如果启用标准 GRPO 的方差归一化，可以使用：

\[
A_{\text{sub}}(\tau,p)
=\frac{R(\tau)-\mu(p)}{\sigma(p)+\epsilon}
\]

均值减法是核心设计，标准差归一化由 trainer 配置决定。

### 3.3 Credit 应用范围

同一 trajectory 中当前 prefix 对应的 SubAgent response 使用该 credit，包括：

- 工作内容；
- `<search>` 内容；
- `<answer>` 内容；
- `<handoff>CONTINUE</handoff>`；
- `<handoff>HANDOFF</handoff>`。

如果同一 prefix 下有多个 `CONTINUE` response，可以让它们共享该
trajectory-level credit。为了避免调用次数多的 trajectory 在 loss 中被重复放大，
可以额外使用当前 prefix 下的 response 数量进行 loss weighting。

### 3.4 示例

以下 trajectories 都经过 prefix：

```text
SubAgent 1 → SubAgent 2
```

最终 reward 为：

```text
1, 0, 1
```

则：

```text
mean reward = 2 / 3
```

对应的 SubAgent credit 为：

```text
1 - 2/3 =  1/3
0 - 2/3 = -2/3
1 - 2/3 =  1/3
```

另一个 prefix，例如：

```text
SubAgent 1 → SubAgent 3
```

应当使用自己的 prefix group，不能和
`SubAgent 1 → SubAgent 2` 混合。

## 4. Router credit

Router credit 是 HANDOFF 节点上的 task-specific route advantage。

### 4.1 Router 分组

Router 在 HANDOFF 后选择下一个 Agent。对于 parent prefix `p`，Router 的分组
使用：

```python
router_group_key = (uid, parent_prefix_key)
```

Router 必须带 `uid`，因为它比较的是同一个问题、同一个 rollout group、同一个
HANDOFF 状态下不同 SubAgent 的后续表现。不同 `uid` 的任务难度和当前状态可能
不同，不能直接混合计算 Router baseline。

### 4.2 Child prefix 的未来表现

假设当前 parent prefix 是 `p`，Router 的候选 Agent 是 `j`。选择 `j` 后形成
child prefix `p+j`。在当前 `uid` 和当前 parent prefix `p` 下，筛选出实际选择了
Agent `j` 的多条 rollout，组成该 child prefix group：

\[
S_{u,p,j}
=\operatorname{mean}\{R(\tau):
\operatorname{uid}(\tau)=u,\
\operatorname{parent\_prefix}(\tau)=p,\
\operatorname{next\_agent}(\tau)=j\}
\]

`S_{u,p,j}` 表示在当前任务和当前 HANDOFF prefix 下选择 Agent `j` 的平均未来
成功率或平均最终 reward。

因此，`S_{u,p,j}` 的计算条件明确为：

```text
uid == u
parent_prefix_key == p
next selected agent == j
```

然后对这些 rollout 的最终 reward 求平均。多条样本通常来自同一个问题的
rollout group；HANDOFF 节点的真实分叉只是产生这些样本的一种方式，并不是
Router credit 公式的另一种分组规则。不同 `uid` 的 trajectory 不能混入同一个
Router group。

如果当前 `(uid, p, j)` 下没有足够的 rollout，`S_{u,p,j}` 的估计会很不稳定；
trainer 应按照配置决定如何处理该 Router 节点，而不能默认混入其他 `uid` 的
reward。

### 4.3 Router 选择概率和 credit 公式

Router 需要提供对每个候选 Agent 的选择概率：

```text
π(j | u, p)
```

该概率应来自 Router 的策略分布，而不是简单统计 batch 中各 Agent 被选中的次数。

按照 Router 自身选择分布得到的期望未来 reward 为：

\[
V_{\text{router}}(u,p)
=\sum_j\pi(j\mid u,p)S_{u,p,j}
\]

如果 Router 实际选择了 Agent `a`，则：

\[
A_{\text{router}}(u,p,a)
=S_{u,p,a}-V_{\text{router}}(u,p)
\]

这表示当前选择的 SubAgent 的未来平均表现是否优于 Router 按自身策略得到的
期望表现。

### 4.4 示例

当前 Router 节点的 parent prefix 为：

```text
SubAgent 1 → SubAgent 2
```

同一个 `uid` 下三个 child prefix 的未来平均 reward 为：

```text
SubAgent 1 → SubAgent 2 → SubAgent 1 : 0.4
SubAgent 1 → SubAgent 2 → SubAgent 2 : 0.8
SubAgent 1 → SubAgent 2 → SubAgent 3 : 0.2
```

Router 概率为：

```text
SubAgent 1: 0.2
SubAgent 2: 0.5
SubAgent 3: 0.3
```

因此：

```text
V_router = 0.2 × 0.4 + 0.5 × 0.8 + 0.3 × 0.2
         = 0.54
```

如果 Router 选择了 SubAgent 2：

```text
router_credit = 0.8 - 0.54 = 0.26
```

这个 credit 用于更新 Router 的 response，不用于更新 SubAgent2 的 response。

### 4.5 FINISH

`FINISH` 可以看作特殊的 Router child action。其 `S` 值是提交当前结果后的未来
成功率，Router expectation 中可以包含：

```text
π(FINISH | u, p) × S(u, p, FINISH)
```

在没有任何可提交结果时，`FINISH` 是无效动作，应按照 invalid action 处理，不能
作为有效 Router credit 样本。

## 5. 两种 credit 的区别

```text
SubAgent credit:
    group key = prefix_key
    不要求 uid-specific
    当前 trajectory reward - 当前 prefix group 平均 reward

Router credit:
    group key = (uid, parent_prefix_key)
    必须 uid-specific
    当前 child prefix 平均 reward
      - Router 按选择概率加权的期望 reward
```

同一个 child prefix group 的平均 reward 同时有两个用途：

1. 它是该 prefix 下 SubAgent credit 的 baseline；
2. 在其 parent HANDOFF 节点上，它是 Router 选择该 SubAgent 的 action value。

但是两种 credit 的分组范围不同：SubAgent 可以跨 `uid` 共享 prefix group，
Router 必须在当前 `uid` 内比较候选 child prefix。

## 6. Trainer 数据和流程

每个 Agent response 至少需要保存：

```text
agent_id
uid
traj_uid
agent_path
prefix_key
parent_prefix_key
handoff_node_id
selected_agent
route_probabilities
final_reward
subagent_credit
router_credit
```

Trainer 的处理顺序是：

```text
收集 rollout
  ↓
根据 HANDOFF 和 Router 选择构造 agent_path/prefix_key
  ↓
按 (traj_uid, prefix_key) 去重 trajectory reward
  ↓
计算 SubAgent prefix group baseline
  ↓
按 (uid, parent_prefix_key) 计算 Router child-prefix value
  ↓
计算 Router credit 和 SubAgent credit
  ↓
写入 token-level reward 或 advantage
  ↓
按 agent/wg_id 拆分 batch
  ↓
分别更新 Router 和对应 SubAgent
```

Router response 使用 `router_credit`，SubAgent response 使用对应 prefix 下的
`subagent_credit`。不同 prefix 的样本可以共同更新同一个 SubAgent 模型，但不能
混用它们的 baseline。

当前 Router Orchestra 已在每个 response batch 行保存：

```text
agent_path
prefix_key
parent_prefix_key
route_probabilities
```

路径状态跨 `run()` 保留，并在 `reset()` 时清空。`CONTINUE` 不追加路径，
Router 选择 worker 时追加一次，`FINISH` 不追加。Router 的候选概率由完整候选
completion 的 log-likelihood 做 softmax；如果 rollout backend 没有候选打分接口，
对应字段为空映射，trainer 不应把它解释为 one-hot 概率。两类 credit 仍由后续
trainer 阶段计算。
