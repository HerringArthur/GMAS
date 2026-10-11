"""Parse Search observations and adapt them to the common context builder."""

import json
import re

from agent_system.agent.agents.router.context import build_context_observation


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


def build_search_observation(agent, env_obs, team_context, step, instructions, suffix='',
                             agent_active_mask=None, collaboration_context=None):
    return build_context_observation(
        agent, env_obs, team_context, step, instructions, suffix,
        agent_active_mask=agent_active_mask, collaboration_context=collaboration_context,
        parse_observation=split_observation,
        environment_budget=search_budget_instruction(agent.config, step, agent.name == 'Router Agent'),
    )
