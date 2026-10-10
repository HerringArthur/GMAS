agent.agent_id
agent.model_id
orchestrator_type都属于config
vanilla_multi_turn_loop的多agent实现，对orchestra搭建有要求self.multiagent_orchestra.run
注意team_context
team_context中是team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
要写入AgentRegistry

• “按照 (traj_uid, agent_id) 聚合”指的是：把同一条完整轨迹中、同一个 SubAgent 产生的多次
  response 看作一个逻辑样本。

  traj_uid 标识一条完整轨迹，agent_id 标识哪个 Agent。

  例如原始 rollout 数据如下：

   traj_uid    agent_id      调用次数    最终 reward
  ━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━━━
   T1          SubAgent 1        2 次              1
  ──────────  ────────────  ──────────  ─────────────
   T1          SubAgent 2        1 次              1
  ──────────  ────────────  ──────────  ─────────────
   T2          SubAgent 1        1 次              0
  ──────────  ────────────  ──────────  ─────────────
   T3          SubAgent 2        2 次              1

  聚合后变成：

  (T1, SubAgent 1) → reward 1
  (T1, SubAgent 2) → reward 1
  (T2, SubAgent 1) → reward 0
  (T3, SubAgent 2) → reward 1

  注意，T1 中 SubAgent 1 调用了两次，但仍然只算一个 (traj_uid, agent_id) 样本，不能让它因
  为调用两次而在平均 reward 中出现两次。

  然后分别计算每个 SubAgent 的平均 reward：

  SubAgent 1:
      (T1, 1), (T2, 0)
      平均 reward = (1 + 0) / 2 = 0.5

  SubAgent 2:
      (T1, 1), (T3, 1)
      平均 reward = (1 + 1) / 2 = 1.0

  于是 SubAgent credit 为：

  (T1, SubAgent 1): 1 - 0.5 =  0.5
  (T2, SubAgent 1): 0 - 0.5 = -0.5

  (T1, SubAgent 2): 1 - 1.0 =  0
  (T3, SubAgent 2): 1 - 1.0 =  0

  最后，这个聚合后的 credit 可以映射回原始 response：

  T1 的 SubAgent 1 第一次调用 → credit 0.5
  T1 的 SubAgent 1 第二次调用 → credit 0.5
  T1 的 SubAgent 2 调用       → credit 0

  这里的聚合主要用于计算平均 reward，避免同一轨迹中重复调用某个 Agent 导致统计偏置。实际
  训练时，如果不希望调用次数多的轨迹产生更大梯度，还可以再给同一 (traj_uid, agent_id) 下
  的多个 response 加上 1 / 调用次数 的 loss weight。

  另外，uid 和 traj_uid 的作用不同：

  - traj_uid：一条具体 rollout 轨迹；
  - uid：同一个问题对应的多条 rollout 轨迹分组。

  所以实际计算通常是：

  先按 (traj_uid, agent_id) 去重并得到每条轨迹的 reward
  再按 (uid, agent_id) 计算该 SubAgent 的平均 reward

  在相同 Agent 路径下，从多次 rollout 汇总结果”与“从同一份完整上下文克隆状态，再分叉执行不同 Agent”是两种采样方式，当前使用的是前者
  无效输出是否另外给予负向训练信号，需要单独设计
  一条trajectory要进入多次训练，因为会有很多分岔点，现在的想法是一条trajectory可以被分入不同的组中