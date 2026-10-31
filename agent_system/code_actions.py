"""Single-call action protocol. Parse Python syntax without executing model code.

Generation stays unconstrained. Parsed arguments and rendered environment actions
are metadata; the original response tokens remain the training target.
"""

import ast
from dataclasses import dataclass
import re


class ActionError(ValueError):
    """An invalid function call, suitable for feedback to the policy."""


@dataclass(frozen=True)
class ParsedAction:
    name: str
    arguments: dict


ROUTER_CALL_FORMAT = '''# Function Call Output
Output exactly one Python function call, using literal keyword arguments:
- call_agent(role="ROLE_NAME", task="The selected role's next task")
- finish()
The task argument is optional. Use an exact available role name. finish() has
no arguments and requires a saved candidate. Do not output XML tags, Markdown
fences, assignments, imports, nested calls, or additional statements.
'''


def action_schema(role, scene):
    """Map function names to required and optional literal keyword arguments."""
    if scene not in ("search", "math"):
        raise ValueError(f"Unsupported scene: {scene}")
    if role == "Router Agent":
        return {"call_agent": ({"role": str}, {"task": str}), "finish": ({}, {})}
    if scene == "search" and role == "SEARCH":
        return {"search": ({"query": str}, {})}
    if role == "VERIFY":
        verdict = "sufficient" if scene == "search" else "approved"
        return {"record_review": ({verdict: bool, "feedback": str}, {})}
    if scene == "search" and role == "ANSWER":
        return {"save_candidate": ({"answer": str}, {"explanation": str})}
    if scene == "math" and role in ("SOLVE", "REVISE"):
        return {"save_candidate": ({"answer": str, "solution": str}, {})}
    raise ValueError(f"Unsupported role {role!r} for {scene}")


def action_instructions(role, scene):
    schema = action_schema(role, scene)
    signatures = []
    for name, (required, optional) in schema.items():
        params = [f"{key}: {kind.__name__}" for key, kind in required.items()]
        params += [f'{key}: {kind.__name__} = ""' for key, kind in optional.items()]
        signatures.append(f"- {name}({', '.join(params)})")
    return (
        "\n# Function Call Output\n"
        "Output exactly one Python function call using keyword arguments and literal "
        "strings or booleans. Multiline calls and triple-quoted strings are allowed. "
        "Use raw strings for LaTeX backslashes. Do not output Markdown fences, XML "
        "tags, imports, assignments, nested calls, or additional statements. "
        "Put your work in the function arguments. Available functions:\n"
        + "\n".join(signatures) + "\n"
    )


def parse_action(source, role, scene, available_roles=()):
    """Accept exactly one named call with typed literal keyword arguments."""
    schema = action_schema(role, scene)
    try:
        tree = ast.parse(source.strip(), mode="exec")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ActionError("Invalid Python syntax; output one complete function call.") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ActionError("Output exactly one function call, without other statements.")
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ActionError("Use a direct function call with literal keyword arguments.")
    name = call.func.id
    if name not in schema:
        raise ActionError(f"{role} can call {', '.join(schema)}; {name} is unavailable.")
    if call.args:
        raise ActionError(f"{name} requires keyword arguments.")
    required, optional = schema[name]
    fields = {**required, **optional}
    arguments = {}
    for keyword in call.keywords:
        key = keyword.arg
        if key not in fields or key in arguments:
            raise ActionError(f"Unknown, repeated, or expanded argument in {name}.")
        value = keyword.value
        if not isinstance(value, ast.Constant) or type(value.value) is not fields[key]:
            raise ActionError(f"{name}.{key} must be a literal {fields[key].__name__}.")
        arguments[key] = value.value
    missing = required.keys() - arguments.keys()
    if missing:
        raise ActionError(f"{name} requires: {', '.join(sorted(missing))}.")
    for key, kind in required.items():
        if kind is str and not arguments[key].strip():
            raise ActionError(f"{name}.{key} must be nonempty.")
    if name == "call_agent" and arguments["role"] not in available_roles:
        raise ActionError(f"Unavailable role {arguments['role']!r}; choose from {', '.join(available_roles)}.")
    # The legacy environment transport uses these tags as control delimiters.
    # Do not let argument text introduce another action during serialization.
    if name in ("search", "save_candidate") and scene == "search":
        payload = arguments["query" if name == "search" else "answer"]
        if re.search(r"</?(?:search|answer|information|think|route)\b", payload, re.I):
            raise ActionError("Action content must not contain environment control tags.")
    if name == "save_candidate" and scene == "math":
        depth = 0
        for char in arguments["answer"]:
            depth += (char == "{") - (char == "}")
            if depth < 0:
                break
        if depth != 0:
            raise ActionError("The mathematical answer must have balanced braces.")
    return ParsedAction(name, arguments)


def render_action(action, scene, source):
    """Render execution/context data, never replacing sampled response tokens."""
    args = action.arguments
    if action.name == "search":
        return f"<search>{args['query'].strip()}</search>"
    if action.name == "save_candidate":
        if scene == "search":
            return f"<answer>{args['answer'].strip()}</answer>"
        return args["solution"].strip() + "\n\\boxed{" + args["answer"].strip() + "}"
    if action.name == "record_review":
        key = "sufficient" if scene == "search" else "approved"
        return f"{key}={args[key]}\nFeedback: {args['feedback']}"
    return source


def project_code_actions(batch, responses, active_mask, *, role, scene, step, available_roles=()):
    """Attach row-aligned protocol metadata and return rendered action strings."""
    import numpy as np

    size = len(responses)
    metadata = batch.non_tensor_batch
    for field in ("action_name", "action_error", "selected_role", "routing_instruction", "execution_status",
                  "environment_action", "environment_result"):
        metadata[field] = np.full(size, "", dtype=object)
    metadata["action_args"] = np.empty(size, dtype=object)
    metadata["action_args"][:] = [dict() for _ in responses]
    metadata["is_action_valid"] = np.zeros(size, dtype=bool)
    metadata["search_executable"] = np.zeros(size, dtype=bool)
    metadata["env_step"] = np.full(size, step, dtype=object)
    rendered = [""] * size
    for i, (source, active) in enumerate(zip(responses, active_mask)):
        if not active:
            continue
        try:
            action = parse_action(source, role, scene, available_roles)
        except ActionError as error:
            metadata["action_error"][i] = str(error)
            metadata["execution_status"][i] = "parse_error"
            continue
        metadata["action_name"][i] = action.name
        metadata["action_args"][i] = action.arguments
        metadata["is_action_valid"][i] = True
        metadata["execution_status"][i] = "parsed"
        metadata["search_executable"][i] = action.name == "search"
        if action.name == "call_agent":
            metadata["selected_role"][i] = action.arguments["role"]
            metadata["routing_instruction"][i] = action.arguments.get("task", "")
        elif action.name == "finish":
            metadata["selected_role"][i] = "FINISH"
        rendered[i] = render_action(action, scene, source)
    return batch, rendered
