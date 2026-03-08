from __future__ import annotations

from typing import Any, Callable, Dict, Optional


SkillInvoker = Callable[[str, Dict[str, Any]], Dict[str, Any]]

_SKILL_INVOKER: Optional[SkillInvoker] = None


def register_skill_invoker(invoker: SkillInvoker) -> None:
    global _SKILL_INVOKER
    _SKILL_INVOKER = invoker


def get_skill_invoker() -> Optional[SkillInvoker]:
    return _SKILL_INVOKER
