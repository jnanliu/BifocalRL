"""
Heuristic classifier for code vs natural language in model generations.

Used to statistics: how often each model uses code vs natural language to solve
problems, and the corresponding accuracy for each approach.
"""

from __future__ import annotations

import re
from typing import Literal

CodeVsNL = Literal["code", "natural_language"]


# Patterns that strongly indicate code usage (Python, etc.)
_CODE_PATTERNS = [
    # Code block markers
    r"```\s*python\b",
    r"```\s*python3\b",
    r"```\s*javascript\b",
    r"```\s*json\b",
    r"```\s*bash\b",
    r"```\s*sh\b",
    # Python keywords (with word boundaries to avoid false positives like "definition")
    r"\bdef\s+\w+\s*\(",
    r"\bimport\s+\w+",
    r"\bfrom\s+\w+\s+import\b",
    r"\bprint\s*\(",
    r"\breturn\s+",
    r"\bfor\s+\w+\s+in\b",
    r"\bwhile\s+",
    r"\bif\s+__name__\s*==",
    r"\bclass\s+\w+\s*:",
    r"\btry\s*:",
    r"\bexcept\s+",
    r"\bwith\s+\w+\s+as\b",
    r"\blambda\s+",
    r"\belif\s+",
    # Common code constructs
    r"\.append\s*\(",
    r"\.split\s*\(",
    r"range\s*\(",
    r"len\s*\(",
    r"int\s*\(",
    r"float\s*\(",
    r"str\s*\(",
    r"list\s*\(",
    r"dict\s*\(",
    r"assert\s+",
    r"==\s*\[",  # list literal in assertion
]

_CODE_RE = re.compile("|".join(f"({p})" for p in _CODE_PATTERNS), re.IGNORECASE | re.DOTALL)


def classify_generation(text: str) -> CodeVsNL:
    """
    Classify a model generation as primarily "code" or "natural_language".

    Heuristic: if the text contains substantial code patterns (code blocks,
    Python keywords, etc.), classify as "code". Otherwise "natural_language".

    Args:
        text: Raw model generation string.

    Returns:
        "code" or "natural_language"
    """
    if not text or not isinstance(text, str):
        return "natural_language"

    s = text.strip()
    if len(s) < 20:
        return "natural_language"

    # Count code pattern matches
    matches = _CODE_RE.findall(s)
    match_count = len(matches) if matches else 0

    # Require at least 2 strong code indicators, or one code block
    if "```" in s and ("python" in s.lower() or "def " in s or "import " in s):
        return "code"
    if match_count >= 2:
        return "code"
    if match_count >= 1 and len(s) > 200:
        # Single indicator in long text: check if it's in a code block
        if "```" in s:
            return "code"
        # Single def/import in long reasoning might be incidental
        return "natural_language"

    return "natural_language"
