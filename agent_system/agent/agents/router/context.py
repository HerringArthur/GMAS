"""Render bounded collaboration inputs from recorded state and source excerpts."""

from collections import OrderedDict
from dataclasses import dataclass
import re
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WorkRecord:
    turn: int
    env_step: int
    author: str
    kind: str
    valid: bool
    response: str


@dataclass(frozen=True)
class CollaborationContext:
    candidate_action: str = ''
    candidate_record: Optional[WorkRecord] = None
    candidate_repeat_count: int = 0
    assigned_worker: str = ''
    routing_instruction: str = ''
    assignment_turn: int = 0
    records: Tuple[WorkRecord, ...] = ()
    loop_round: int = 0
    loop_limit: int = 0
    candidate_evidence: Tuple[Tuple[str, str], ...] = ()
    candidate_quality: str = 'none'
    candidate_review: str = 'unreviewed'
    last_worker_decision: str = ''


class ContextTokenCache:
    """Bound tokenization caches to one agent rollout, across both environments."""

    def __init__(self, tokenizer, max_entries=2048, max_tokens=262144, max_chars=1048576):
        self.tokenizer = tokenizer
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.entries = OrderedDict()
        self.num_tokens = 0
        self.num_chars = 0

    def encode(self, text):
        if text in self.entries:
            self.entries.move_to_end(text)
            return self.entries[text]
        tokens = tuple(self.tokenizer.encode(text, add_special_tokens=False)) if self.tokenizer is not None else tuple(text)
        if len(tokens) > self.max_tokens or len(text) > self.max_chars:
            return tokens
        self.entries[text] = tokens
        self.num_tokens += len(tokens)
        self.num_chars += len(text)
        while (len(self.entries) > self.max_entries or self.num_tokens > self.max_tokens
               or self.num_chars > self.max_chars):
            old_text, old_tokens = self.entries.popitem(last=False)
            self.num_tokens -= len(old_tokens)
            self.num_chars -= len(old_text)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False) if self.tokenizer is not None else ''.join(tokens)


OMITTED = '[omitted for length]'


def excerpt(text, budget, cache):
    """Keep source text from both ends; explicitly mark every omitted middle."""
    tokens = cache.encode(text)
    if len(tokens) <= budget:
        return text
    marker = '\n[... omitted ...]\n'
    kept = max(0, budget - len(cache.encode(marker)))
    while kept > 0:
        head = (kept + 1) // 2
        tail = kept // 2
        prefix = cache.decode(tokens[:head])
        suffix = cache.decode(tokens[-tail:]) if tail else ''
        # A token boundary can split a Unicode character. Keep actual source
        # substrings rather than introducing replacement characters in excerpts.
        while prefix and not text.startswith(prefix):
            prefix = prefix[:-1]
        while suffix and not text.endswith(suffix):
            suffix = suffix[1:]
        result = prefix + marker + suffix
        overflow = len(cache.encode(result)) - budget
        if overflow <= 0:
            return result
        kept -= max(1, overflow)
    return OMITTED if len(cache.encode(OMITTED)) <= budget else ''


def allocate_budgets(minimums, demands, weights, extra):
    """Share surplus space, redistributing the share of sources that already fit."""
    budgets = list(minimums)
    while extra > 0:
        pending = [i for i in range(len(budgets)) if budgets[i] < demands[i]]
        if not pending:
            break
        total_weight = sum(weights[i] for i in pending)
        available = extra
        for i in pending:
            add = min(extra, demands[i] - budgets[i], max(1, available * weights[i] // total_weight))
            budgets[i] += add
            extra -= add
    return budgets


def pack_sources(sources, budget, cache):
    """Retain recent distinct sources with intact attribution and bounded excerpts.

    Sources are chronological (label, text, deduplication key) tuples. Repeated
    source text is represented by its most recent occurrence, never synthesized.
    """
    unique = []
    seen = set()
    for label, body, key in reversed(sources):
        if key not in seen:
            unique.append((label, body))
            seen.add(key)
    unique.reverse()
    all_text = '\n\n'.join(label + '\n' + body for label, body in unique)
    duplicate_note = '[Repeated records omitted.]\n' if len(unique) < len(sources) else ''
    if len(cache.encode(duplicate_note + all_text)) <= budget:
        return duplicate_note + all_text
    note = '[Source excerpts; some content omitted.]\n'
    selected = []
    for source in reversed(unique):
        proposed = [source] + selected
        # Reserve useful body space before adding older headers. Otherwise a
        # long history can spend its whole budget on attribution with no work.
        minimum = note + '\n\n'.join(label + '\n' + excerpt(body, 64, cache) for label, body in proposed)
        if len(cache.encode(minimum)) > budget:
            break
        selected = proposed
    if not selected:
        return OMITTED if len(cache.encode(OMITTED)) <= budget else ''
    minimum = note + '\n\n'.join(label + '\n' + excerpt(body, 64, cache) for label, body in selected)
    extra = max(0, budget - len(cache.encode(minimum)) - 8)
    floors = [len(cache.encode(excerpt(body, 64, cache))) for _, body in selected]
    demands = [max(floors[i], len(cache.encode(body))) for i, (_, body) in enumerate(selected)]
    while True:
        budgets = allocate_budgets(floors, demands, [1] * len(selected), extra)
        result = note + '\n\n'.join(
            label + '\n' + excerpt(body, allowance, cache)
            for (label, body), allowance in zip(selected, budgets)
        )
        overflow = len(cache.encode(result)) - budget
        if overflow <= 0:
            return result
        if extra == 0:
            return minimum
        extra = max(0, extra - overflow - 8)


def record_label(record):
    validity = 'protocol valid' if record.valid else 'protocol invalid'
    return f'Turn {record.turn}, step {record.env_step}, {record.author}, {record.kind}, {validity}'


def candidate_work(state):
    """Associate source work with the saved candidate, without rescanning history."""
    record = state.candidate_record
    if record is None:
        return ''
    response = re.sub(r'<handoff>\s*(?:CONTINUE|HANDOFF)\s*</handoff>\s*$', '', record.response).strip()
    reasoning = re.findall(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    visible = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    if state.candidate_action.startswith('<answer>'):
        visible = re.sub(r'<answer>.*?</answer>', '', visible, flags=re.DOTALL)
    elif state.candidate_action in visible:
        before, _, after = visible.rpartition(state.candidate_action)
        visible = before + after
    return '\n'.join(part.strip() for part in reasoning + [visible] if part.strip())


def collaboration_sections(state, agent_name, fallback_history):
    """Return required state plus optional attributed source groups."""
    core = []
    groups = []
    if state is None:
        core.append('# Candidate answer\nNo stored candidate state supplied.')
        core.append('# Current assignment\nNo structured assignment supplied.')
        if fallback_history.strip():
            groups.append(('# Historical work (unverified)', [
                ('Raw shared history; candidate status unavailable', fallback_history, fallback_history)
            ], 1))
        return core, groups

    if agent_name == 'Router Agent':
        title = '# Previous assignment (historical)'
    else:
        title = '# Current assignment'
    if state.assigned_worker and (agent_name == 'Router Agent' or state.assigned_worker == agent_name):
        task = state.routing_instruction or 'No specific task instruction was provided for this assignment.'
        core.append(f'{title}\nWorker: {state.assigned_worker}; routing turn: {state.assignment_turn}\n{task}')
    else:
        message = 'No worker assignment has been recorded.' if agent_name == 'Router Agent' else 'No current assignment for this worker.'
        core.append(title + '\n' + message)

    if agent_name == 'Router Agent' and state.last_worker_decision:
        previous_worker = state.assigned_worker or 'the previous worker'
        core.append(
            '# Previous worker control\n'
            f'{previous_worker} returned <handoff>{state.last_worker_decision}</handoff>. '
            'Every valid worker response is reviewed by the Router before another worker turn. '
            + (
                'A candidate exists; prefer a different worker for independent verification when one is available.'
                if state.last_worker_decision == 'HANDOFF' and state.candidate_action
                else 'Decide whether to finish, continue verification, or assign the next worker.'
            )
        )

    if state.candidate_action:
        record = state.candidate_record
        if record is None:
            raise ValueError('A saved candidate must include its source record.')
        core.append('# Candidate answer (awaiting review)\n'
                    f'Author: {record.author}; turn: {record.turn}; step: {record.env_step}\n'
                    f'Consecutive occurrences of this answer text: {state.candidate_repeat_count}\n'
                    f'Heuristic evidence status: {state.candidate_quality}\n'
                    f'Router review status: {state.candidate_review}\n'
                    'Candidate (verbatim):\n' + state.candidate_action)
        support = candidate_work(state)
        if support:
            # Keep the candidate's own reasoning ahead of older history when
            # the prompt budget is tight.  This is the most relevant support
            # for the Router's quality review; historical records can be
            # abbreviated first.
            groups.append(('# Work accompanying the candidate (source excerpts, unverified)', [
                (record_label(record), support, (record.author, support))
            ], 5))
        else:
            core.append('# Work accompanying the candidate\nNo accompanying reasoning was supplied.')
        if state.candidate_evidence:
            groups.append(('# Evidence attached to candidate (review this support)', [
                (f'Candidate evidence {index + 1}', f'Query: {query}\n{result}', (query, result))
                for index, (query, result) in enumerate(state.candidate_evidence)
            ], 4))
        else:
            core.append('# Evidence attached to candidate\nNo retrieved evidence was captured.')
    else:
        core.append('# Candidate answer (awaiting review)\nNo candidate answer has been saved.')

    # Invalid generations stay in WorkRecord for diagnostics and PPO labels,
    # but must not be presented as demonstrations in the next prompt.  The
    # correction feedback section already tells the same agent what to fix.
    records = [record for record in state.records
               if record.valid
               and record.turn != state.assignment_turn
               and (state.candidate_record is None or record.turn != state.candidate_record.turn)]
    if records:
        groups.append(('# Recent work (historical; validity refers to protocol, not correctness)', [
            (record_label(record), record.response, (record.author, record.kind, record.valid, record.response))
            for record in records
        ], 1))
    else:
        core.append('# Recent work\nNo other work has been recorded.')

    if state.loop_limit:
        core.append('# Orchestration budget\n'
                    f'Round {state.loop_round} of {state.loop_limit} in this invocation; '
                    f'{max(0, state.loop_limit - state.loop_round)} further rounds remain after this round. '
                    'A round can include a router call and a worker call. This budget is separate '
                    'from environment steps and resets on the next orchestration invocation.')
    return core, groups


def build_context_observation(agent, env_obs, team_context, step, instructions, suffix='',
                              agent_active_mask=None, collaboration_context=None,
                              parse_observation=None, environment_budget=''):
    """Build Math/Search inputs with protected state and shared excerpt budgeting."""
    batch_size = len(env_obs['text'])
    mask = np.ones(batch_size, dtype=bool) if agent_active_mask is None else np.asarray(agent_active_mask, dtype=bool)
    if mask.shape != (batch_size,):
        raise ValueError('agent_active_mask must have one entry per observation.')
    if collaboration_context is not None and len(collaboration_context) != batch_size:
        raise ValueError('collaboration_context must have one entry per observation.')
    obs = dict(env_obs)
    obs['text'] = [''] * batch_size
    limit = getattr(getattr(agent.config, 'data', None), 'max_prompt_length', 4096) - 256
    cache = getattr(agent, '_context_token_cache', None)
    if cache is None or cache.tokenizer is not agent.tokenizer:
        cache = agent._context_token_cache = ContextTokenCache(agent.tokenizer)
    for i in np.flatnonzero(mask):
        question, evidence = parse_observation(env_obs['text'][i]) if parse_observation else (env_obs['text'][i], None)
        state = collaboration_context[i] if collaboration_context is not None else None
        if collaboration_context is not None and state is None:
            raise ValueError('An active sample is missing its collaboration context.')
        core, groups = collaboration_sections(state, agent.name, team_context[i])
        core.insert(0, '# Original task\n' + question)
        if evidence is not None:
            if evidence:
                groups.append(('# Retrieved evidence (source excerpts)', [
                    (f'Query {j + 1}', f'Query: {query}\n{result}', (query, result))
                    for j, (query, result) in enumerate(evidence)
                ], 2))
            else:
                core.append('# Retrieved evidence\nNo search results yet.')
        if not environment_budget:
            max_steps = getattr(agent.config.env, 'max_steps', None)
            budget_text = f'# Environment budget\nEnvironment step {step}'
            budget_text += f' of {max_steps}.' if max_steps is not None else '.'
        else:
            budget_text = environment_budget.strip()
        tail = [instructions.strip(), suffix.strip(), budget_text]
        feedback = getattr(agent, 'search_feedback', None)
        if feedback is not None and i < len(feedback) and feedback[i]:
            tail.append('# Correction to your previous response\n' + feedback[i])

        def assemble(bodies):
            excerpts = [header + '\n' + body for (header, _, _), body in zip(groups, bodies)]
            return '\n\n'.join(part for part in core + excerpts + tail if part)

        minimum = assemble([OMITTED] * len(groups))
        if len(cache.encode(minimum)) > limit:
            raise ValueError('Core collaboration context exceeds max_prompt_length; increase the prompt budget. '
                             'The task, assignment, candidate, feedback, and output rules will not be truncated.')
        extra = max(0, limit - len(cache.encode(minimum)) - 32)
        floors = [len(cache.encode(OMITTED))] * len(groups)
        demands = [max(floors[j], len(cache.encode(pack_sources(sources, limit, cache))))
                   for j, (_, sources, _) in enumerate(groups)]
        while True:
            budgets = allocate_budgets(floors, demands, [weight for _, _, weight in groups], extra)
            bodies = [pack_sources(sources, allowance, cache)
                      for (_, sources, _), allowance in zip(groups, budgets)]
            prompt = assemble(bodies)
            overflow = len(cache.encode(prompt)) - limit
            if overflow <= 0:
                break
            if extra == 0:
                prompt = minimum
                break
            extra = max(0, extra - overflow - 32)
        obs['text'][i] = prompt
    return obs


def ensure_prompt_fits(agent, gen_batch, obs, active_mask):
    """Prevent downstream truncation when a system prompt/template exceeds the reserve."""
    if agent.tokenizer is None or not hasattr(agent.tokenizer, 'apply_chat_template'):
        return
    data_config = agent.config.data
    raw_prompts = gen_batch.non_tensor_batch.get('raw_prompt')
    for i in np.flatnonzero(active_mask):
        system = 'You are a helpful and harmless assistant.'
        if raw_prompts is not None:
            for message in raw_prompts[i]:
                if message['role'] == 'system':
                    system = message['content']
        text = agent.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': obs['text'][i]}],
            add_generation_prompt=True, tokenize=False,
            **getattr(data_config, 'apply_chat_template_kwargs', {}),
        )
        if len(agent.tokenizer.encode(text, add_special_tokens=False)) > data_config.max_prompt_length:
            raise ValueError('The complete collaboration prompt exceeds max_prompt_length after applying '
                             'the chat template; increase the prompt budget to avoid truncating core context.')
