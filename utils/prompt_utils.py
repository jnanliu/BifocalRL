# coding=utf-8
"""Prompt rendering and formatting helpers shared across pipelines."""

import copy
from typing import Any, Dict, List

from utils.trl_compat import maybe_apply_chat_template


def messages_to_plain_text(prompt) -> str:
    """Extract plain text from a chat-message list when no chat_template is available.

    For base models without a chat template we must avoid sending the raw
    ``str([{'role': 'system', ...}])`` representation.  Instead, concatenate
    the message contents into a readable plain-text prompt.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: List[str] = []
        for msg in prompt:
            if isinstance(msg, dict):
                c = msg.get("content", "")
                if c is None:
                    continue
                s = str(c).strip()
                if s:
                    parts.append(s)
            else:
                parts.append(str(msg))
        return "\n\n".join(parts)
    return str(prompt)


def render_prompt(
    problem,
    tokenizer,
    has_template: bool,
    *,
    apply_chat_template_kwargs: Dict[str, Any] | None = None,
) -> str:
    """Apply chat template if available, otherwise return raw prompt text.

    IMPORTANT: ``maybe_apply_chat_template`` mutates the input dict in-place
    (e.g. replaces ``problem["prompt"]`` from a message list to a rendered
    string).  We deep-copy *problem* first so that callers can safely call
    ``render_prompt`` multiple times with different tokenizers on the same
    problem dict (critical for cross-model pipelines).
    """
    if not has_template:
        prompt = problem.get("prompt", problem) if isinstance(problem, dict) else problem
        return messages_to_plain_text(prompt)
    try:
        prompt_obj = problem.get("prompt", problem) if isinstance(problem, dict) else problem

        # HACRL-style prompt rendering path: use tokenizer.apply_chat_template directly
        # with explicit kwargs.  Default enable_thinking=False to match HACRL
        # (verl/utils/dataset/rl_dataset.py disables thinking mode for RL training).
        if hasattr(tokenizer, "apply_chat_template") and isinstance(prompt_obj, list):
            kwargs = {"add_generation_prompt": True, "tokenize": False}
            if apply_chat_template_kwargs:
                kwargs.update(apply_chat_template_kwargs)
            if "enable_thinking" not in kwargs:
                kwargs["enable_thinking"] = False
            # Passing enable_thinking=None signals "don't set this kwarg at all"
            # (matches TTRL behaviour where the parameter is never passed).
            if kwargs.get("enable_thinking") is None:
                del kwargs["enable_thinking"]
            return str(tokenizer.apply_chat_template(prompt_obj, **kwargs))

        problem_safe = copy.deepcopy(problem) if isinstance(problem, dict) else problem
        res = maybe_apply_chat_template(problem_safe, tokenizer)
        if isinstance(res, dict):
            if res.get("prompt") is not None:
                return str(res.get("prompt"))
            if res.get("text") is not None:
                return str(res.get("text"))
            return str(problem_safe.get("prompt", ""))
        return str(res)
    except Exception:
        prompt = problem.get("prompt", problem) if isinstance(problem, dict) else problem
        return messages_to_plain_text(prompt)


def render_prompts(
    problems: List[Dict[str, Any]],
    tokenizer,
    *,
    has_template: bool | None = None,
    apply_chat_template_kwargs: Dict[str, Any] | None = None,
) -> List[str]:
    """Render a batch of problems into prompt strings using *tokenizer*'s chat template."""
    if has_template is None:
        has_template = getattr(tokenizer, "chat_template", None) is not None
    return [
        render_prompt(
            p,
            tokenizer,
            has_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        for p in problems
    ]


def flatten_prompts(unique_prompts: List[str], num_generations: int) -> List[str]:
    """Repeat each prompt *num_generations* times to match flattened completions ordering."""
    out: List[str] = []
    for p in unique_prompts:
        out.extend([p] * num_generations)
    return out
