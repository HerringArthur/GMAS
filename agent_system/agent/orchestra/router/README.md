# Router and SubAgents

The credit-assignment design for HANDOFF prefixes, Router decisions, and
SubAgent trajectory groups is documented in
[`CREDIT_ASSIGNMENT.md`](./CREDIT_ASSIGNMENT.md).

`RouterMultiAgentOrchestra` uses one Router Agent and three role-free SubAgents by
default. Agent names and model assignments come from the existing
`agent.agent_ids` / `agent.model_ids` lists. `Router Agent` is reserved for routing;
all other configured names create SubAgents. `FINISH` is a reserved control value.

The `router_trainer.yaml` preset inherits `ppo_trainer.yaml` and selects this team:

```yaml
agent:
  agent_ids: ["Router Agent", "Sub Agent 1", "Sub Agent 2", "Sub Agent 3"]
  model_ids: ["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B"]
  orchestra_type: router
  model_sharing: false
  use_agent_memory: false
  orchestra:
    router:
      max_loop_num: 8
```

Select the preset with `python -m verl.trainer.main_ppo --config-name router_trainer`
and your usual data and training overrides. It defaults to the math environment;
use `env.env_name=search` for a search task. Other team sizes and worker names can
still be configured, with one model entry per agent.

The existing model-sharing setting also works: worker identities remain distinct
even when they share a model/worker group. Configure the environment as usual for
math or search tasks.

## Control flow

1. For each active sample without a current worker, the router receives the task,
   shared progress, and available worker names.
2. `<route>Sub Agent 1</route>` selects that worker. `<route>FINISH</route>` submits
   the latest available worker result and ends collaboration for that sample in
   the current environment step. FINISH before any result exists is invalid.
3. The selected worker completes part of the work and ends its response with
   `<handoff>CONTINUE</handoff>` or `<handoff>HANDOFF</handoff>`.
4. CONTINUE keeps the same worker for the next generation. HANDOFF schedules a
   new router call; it does not itself finish the task. The router may reselect
   the same worker.

The implementation follows Search/Math Orchestra: `run()` explicitly calls an
agent, updates `team_context`, saves its batch, and updates actions with
`update_text_action()`. No separate `_call_agent()` wrapper is used.

Routing and continuation are independent for every batch item. The router's
selection is in `batch.non_tensor_batch["selected_agent"]`; the worker's control
signal is in `batch.non_tensor_batch["handoff_decision"]`. Both agents return
`(batch, full_text_responses)`. Their full outputs, including reasoning and control
tags, are appended to `team_context` with the emitting agent's name after each
call. Shared context and current workers survive environment steps and are cleared
by `reset()` at the beginning of a new rollout.

Each saved response row also carries `agent_path`, `prefix_key`,
and `parent_prefix_key`. A worker selection extends the
path once; CONTINUE, HANDOFF, and FINISH do not append a node. `prefix_key` is the
tuple form of `agent_path`, while `parent_prefix_key` identifies the prefix before
the current worker segment. The Router generates and records its selection without
scoring alternative completions or writing `route_probabilities` to the batch.
The planned trainer computes route frequencies from valid Router selections in
each `(uid, parent_prefix_key)` group after deduplicating trajectories. Valid FINISH
selections count as a candidate alongside workers. Unobserved candidates are
excluded; a group with only one observed candidate has zero Router credit.
Credit computation is not yet implemented.

Unlike Search/Math, `initialize_context()` reuses `self.team_context` on subsequent
calls. For example, a worker's intermediate derivation remains available after a
search even if that derivation was not included in the search query. This memory
is independent of the legacy `agent.use_agent_memory` setting.

Search history is read from `env_obs["text"]`; raw `anchor` values are not appended
to the team context. There is no `awaiting_tools` field. Use the preset's
`agent.use_agent_memory: false` and a positive `env.history_length` so that
`SearchEnvironmentManager` includes tool results in its observation. Disabling
environment history or replacing it with the legacy `{memory}` placeholder does
not supply tool results to this orchestra.

## Environment compatibility and bounds

`<search>` and `<answer>` are task-format compatibility tags, not alternatives to
CONTINUE/HANDOFF. In search environments, a complete search request yields to the
existing rollout loop's `envs.step()` so the tool can execute. After the tool result
arrives, CONTINUE resumes the requesting worker, while HANDOFF invokes the router.
A search request is never a final submission candidate. An answer tag alone does
not stop internal collaboration; the worker must hand off for the router to choose
FINISH. Only the extracted search/answer block is sent to the search environment;
the complete response stays in shared context. Math environments receive worker
text with the trailing handoff tag removed.

Search tasks use dedicated worker and router prompts in both training and
evaluation. Workers are instructed to assess the evidence, emit one search or
answer block outside their reasoning, and put only the concise final value inside
an answer block. Once an answer is ready, the worker should HANDOFF so the router
can review it. The router sees complete routing options and is instructed to
FINISH only when an actual answer block is present and supported by the evidence.
These are generation instructions; the control parser and loop behavior described
above still determine how generated responses execute.

The Search prompt builder keeps the original question, a bounded view of recent
distinct agent work and the most recent proposed answer, then the retrieved
evidence and current instructions. It decodes the retriever's JSON result wrapper
into readable text and shares the evidence budget across retained search steps.
It reserves 256 tokens of `data.max_prompt_length` for chat formatting and the
system message; the unabridged responses remain in `team_context` and rollout
buffers. Very long questions/rules that cannot fit produce an explicit error.

Both agents see the remaining environment budget. For `env.max_steps=4`, only
steps 1–3 can execute searches: SearchEnv terminates before executing a search on
step 4. On the last step, the current worker receives a dedicated final-answer
prompt and is asked to return its best-supported short answer with HANDOFF. The
router still decides FINISH; the prompt does not fabricate an answer or force a
control transition when the model ignores it.

Search worker validity requires both a legal handoff and exactly one nonempty,
complete search/answer action outside reasoning. Missing, multiple, nested, or
incomplete actions are marked invalid and receive specific correction feedback
on that worker's next call. Feedback is separate for every sample, cleared after
a valid response, and reset between rollouts. Invalid outputs remain in the
training buffer with `is_action_valid=False`, including when the control tag was
legal. Retries use the existing internal loop budget.

`max_loop_num` bounds each `run()` call: each sample receives at most one worker
generation per loop, plus a router generation when routing is needed. If this
limit is reached before FINISH, the latest available result is returned as a
fallback. Invalid routes retry the router; invalid worker controls/actions retry the
same worker. Both consume the loop budget. If no usable result is produced, the
action is an empty string, which the environment handles as an invalid submission.
Environment `dones`, not the routing metadata, remain the rollout loop's source of
episode completion.

## Validation

```sh
python3 -m unittest discover -s tests/agent_system -p 'test_router.py' -v
```

These CPU tests exercise the real agent and orchestra logic with model generation,
`DataProto`, and preprocessing mocked. They do not run distributed generation or
training.
