"""Build bounded Search inputs without changing the stored rollout history."""

from collections import OrderedDict
import json
import re

import numpy as np


class _SearchTokenCache:
    """Bound repeated question/rule/evidence tokenization to one agent rollout."""

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
        tokens = tuple(_tokens(text, self.tokenizer))
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


def _tokens(text, tokenizer, cache=None):
    if cache is not None:
        return cache.encode(text)
    return tokenizer.encode(text, add_special_tokens=False) if tokenizer is not None else list(text)


def _clip(text, budget, tokenizer, cache=None):
    tokens = _tokens(text, tokenizer, cache)
    if len(tokens) <= budget:
        return text
    if budget <= 0:
        return ''
    # Leave room for the truncation marker, including tokenizer boundary effects.
    kept = tokens[:max(0, budget - 8)]
    text = tokenizer.decode(list(kept), skip_special_tokens=True) if tokenizer is not None else ''.join(kept)
    return text + ' [...]'


def split_observation(text):
    """Read the Search manager's question and query/result pairs losslessly.

    Unknown observation formats remain intact instead of silently losing evidence.
    JSON decoding applies only to the retriever's result wrapper, not document text.
    """
    question = re.search(r'The question is: (.*?)\nYour team can access', text, re.DOTALL)
    if question is None:
        return text.strip(), []
    history = text.split('History:\n', 1)
    if len(history) == 1:
        return question.group(1).strip(), []
    records = re.findall(
        r'<search>(.*?)</search>\s*<information>(.*?)</information>',
        history[1], flags=re.DOTALL,
    )
    if not records and history[1].strip():
        return question.group(1).strip(), [('Previous interaction', history[1].strip())]
    results = []
    for query, payload in records:
        try:
            decoded = json.loads(payload)
        except (ValueError, TypeError):
            decoded = None
        evidence = decoded['result'] if isinstance(decoded, dict) and isinstance(decoded.get('result'), str) else payload
        record = (query.strip(), evidence.strip())
        if record not in results:
            results.append(record)
    return question.group(1).strip(), results


def compact_progress(context, budget, tokenizer):
    """Keep candidate answers and recent distinct work; raw history stays in logs."""
    parts = re.split(r'\n(?=The output of ")', context.strip())
    unique = []
    for part in reversed(parts):
        if part.strip() and part not in unique:
            unique.append(part)
    unique.reverse()
    candidates = [p for p in unique if re.search(r'<answer>.+?</answer>', re.sub(r'<think>.*?</think>', '', p, flags=re.DOTALL), re.DOTALL)]
    candidate = ''
    if candidates:
        visible = re.sub(r'<think>.*?</think>', '', candidates[-1], flags=re.DOTALL)
        candidate = '\n\nMost recent proposed answer (verbatim):\n' + re.findall(r'<answer>.*?</answer>', visible, re.DOTALL)[-1]
    budget -= len(_tokens(candidate, tokenizer))
    selected = [p for p in unique[-3:] if p not in candidates]
    if not selected:
        return candidate.strip() or 'No previous agent work.'
    per_entry = max(0, budget // len(selected) - 4)
    compact = []
    for part in selected:
        part = re.sub(
            r'<think>(.*?)</think>',
            lambda match: '<think>' + _clip(match.group(1), min(96, per_entry // 2), tokenizer) + '</think>',
            part, flags=re.DOTALL,
        )
        compact.append(_clip(part, per_entry, tokenizer))
    return _clip('\n\n'.join(compact), budget, tokenizer) + candidate


def search_budget_instruction(config, step, is_router=False):
    max_steps = getattr(config.env, 'max_steps', None)
    if max_steps is None:
        return ''
    searches_left = max(0, max_steps - step)
    text = (f'\n\n# Remaining interaction budget\nEnvironment step {step} of {max_steps}. '
            f'At most {searches_left} further search calls can return evidence. '
            'The final environment step must submit an answer; a search on that step will not run.\n')
    if searches_left == 0:
        if is_router:
            text += ('Final step: review the existing candidate using the evidence and your knowledge. '
                     'FINISH if it answers the question and no better-supported correction is needed. '
                     'If no candidate exists or a correction is needed, select a worker to produce '
                     'a final answer from the information already available.')
        else:
            text += ('Final step: use the retrieved evidence and your knowledge to give your '
                     'best-supported answer now, even if some uncertainty remains. Do not search again. '
                     'Put any reasoning inside <think>, then output <answer>short final value</answer>'
                     '<handoff>HANDOFF</handoff> so the router can review and submit it.')
    return text


def build_search_observation(agent, env_obs, team_context, step, instructions, suffix='', agent_active_mask=None):
    """Prioritize the question/rules, then share space among work and tool evidence."""
    batch_size = len(env_obs['text'])
    if agent_active_mask is None:
        indices = range(batch_size)
    else:
        mask = np.asarray(agent_active_mask, dtype=bool)
        if mask.shape != (batch_size,):
            raise ValueError('agent_active_mask must have one entry per observation.')
        indices = np.flatnonzero(mask)
    # Only text is rewritten. Leave inactive slots empty and preserve global row
    # numbers so feedback and the eventual training records keep their identity.
    obs = dict(env_obs)
    obs['text'] = [''] * batch_size
    limit = getattr(getattr(agent.config, 'data', None), 'max_prompt_length', 4096)
    # Reserve tokens for the chat template/system message added by preprocess_batch.
    limit -= 256
    tokenizer = agent.tokenizer
    cache = getattr(agent, '_search_token_cache', None)
    if cache is None or cache.tokenizer is not tokenizer:
        cache = agent._search_token_cache = _SearchTokenCache(tokenizer)
    for i in indices:
        env_text = env_obs['text'][i]
        question, records = split_observation(env_text)
        start = f'# Question\n{question}\n\n# Team progress (prior reasoning may be outdated)\n'
        evidence_header = '\n\n# Retrieved evidence (read this before deciding the next action)\n'
        end = '\n\n' + instructions + search_budget_instruction(agent.config, step, agent.name == 'Router Agent') + suffix
        feedback = getattr(agent, 'search_feedback', None)
        if feedback is not None and i < len(feedback) and feedback[i]:
            end += '\n\n# Correction to your previous response\n' + feedback[i]
        fixed = len(_tokens(start + evidence_header + end, tokenizer, cache))
        if fixed >= limit:
            raise ValueError('Search question and instructions exceed max_prompt_length; increase the prompt budget.')
        remaining = limit - fixed
        progress = compact_progress(team_context[i], min(512, remaining // 4), tokenizer)
        evidence_budget = max(0, remaining - len(_tokens(progress, tokenizer)) - 32)
        if records:
            per_record = max(0, evidence_budget // len(records) - 8)
            evidence = '\n\n'.join(
                _clip(f'Query {j + 1}: {query}\n{result}', per_record, tokenizer, cache)
                for j, (query, result) in enumerate(records)
            )
        else:
            evidence = 'No search results yet.'
        obs['text'][i] = start + progress + evidence_header + evidence + end
        if len(_tokens(obs['text'][i], tokenizer)) > limit:
            raise ValueError('Search context exceeded its reserved prompt budget.')
    return obs
