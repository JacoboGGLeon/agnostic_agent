from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING


if TYPE_CHECKING:
    from agnostic_agent.skills import Skill


@dataclass
class SkillConsistencyReport:
    manifest_valid: bool
    instructions_valid: bool
    tools_resolved: int
    tools_declared: int
    knowledge_resolved: int
    knowledge_declared: int
    planner_policy_valid: bool
    runtime_alignment_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillCapabilityContract:
    skill_name: str
    world: str
    declared_tools: List[str]
    declared_knowledge: List[str]
    declared_intents: List[str]
    declared_entities: List[str]
    intent_entity_requirements: Dict[str, Dict[str, List[str]]]
    planner_policy: Dict[str, Any]
    summarizer_policy: Dict[str, Any]
    validator_policy: Dict[str, Any]
    ui_metadata: Dict[str, Any]
    instructions_summary: str
    instruction_capabilities: Dict[str, Any]
    runtime_capabilities: Dict[str, Any]
    consistency_report: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize_instructions(instructions: str, max_len: int = 360) -> str:
    if not instructions:
        return ""
    lines: List[str] = []
    for raw_line in instructions.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
        if len(" ".join(lines)) >= max_len:
            break
    summary = " ".join(lines).strip()
    if len(summary) > max_len:
        return summary[:max_len].rstrip() + "..."
    return summary


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _runtime_intents_from_skill_source(skill: "Skill") -> List[str]:
    file_path = str(skill.file_path or "").strip()
    if not file_path:
        return []
    manifest_path = Path(file_path)
    skill_path = manifest_path.parent / "skill.py"
    if not skill_path.exists():
        return []
    text = skill_path.read_text(encoding="utf-8", errors="ignore")
    intents: List[str] = []
    for intent in skill.intents:
        if f'"{intent}"' in text or f"'{intent}'" in text:
            intents.append(intent)
    return sorted(set(intents))


def _build_instruction_capabilities(skill: "Skill") -> Dict[str, Any]:
    instructions_norm = _normalize_text(skill.instructions)
    mentioned_tools = [tool for tool in skill.tools if _normalize_text(tool) in instructions_norm]
    mentioned_intents = [intent for intent in skill.intents if _normalize_text(intent.replace("_", " ")) in instructions_norm]
    critical_tools: List[str] = []
    planner_tools = skill.planner_policy.get("intent_to_tools") if isinstance(skill.planner_policy, dict) else {}
    if isinstance(planner_tools, dict):
        for tools_for_intent in planner_tools.values():
            if isinstance(tools_for_intent, list):
                for tool_name in tools_for_intent:
                    normalized = str(tool_name).strip()
                    if normalized and normalized not in critical_tools:
                        critical_tools.append(normalized)
    return {
        "mentions_world": _normalize_text(skill.world or skill.name) in instructions_norm,
        "mentioned_tools": mentioned_tools,
        "mentioned_intents": mentioned_intents,
        "critical_tools": critical_tools,
    }


def _build_runtime_capabilities(skill: "Skill") -> Dict[str, Any]:
    planner_tools = skill.planner_policy.get("intent_to_tools") if isinstance(skill.planner_policy, dict) else {}
    runtime_intents = _runtime_intents_from_skill_source(skill)
    return {
        "entrypoint": skill.entrypoint,
        "entrypoint_exists": bool(skill.entrypoint),
        "planner_intents": sorted(planner_tools.keys()) if isinstance(planner_tools, dict) else [],
        "planner_tools": planner_tools if isinstance(planner_tools, dict) else {},
        "skill_runtime_intents": runtime_intents,
    }


def build_skill_capability_contract(
    skill: "Skill",
    consistency_report: SkillConsistencyReport,
) -> SkillCapabilityContract:
    instruction_capabilities = _build_instruction_capabilities(skill)
    runtime_capabilities = _build_runtime_capabilities(skill)
    return SkillCapabilityContract(
        skill_name=skill.name,
        world=skill.world or skill.name,
        declared_tools=list(skill.tools or []),
        declared_knowledge=list(skill.knowledge or []),
        declared_intents=list(skill.intents or []),
        declared_entities=list(skill.entities or []),
        intent_entity_requirements=dict(skill.intent_entity_requirements or {}),
        planner_policy=dict(skill.planner_policy or {}),
        summarizer_policy=dict(skill.summarizer_policy or {}),
        validator_policy=dict(skill.validator_policy or {}),
        ui_metadata=dict(skill.ui or {}),
        instructions_summary=summarize_instructions(skill.instructions),
        instruction_capabilities=instruction_capabilities,
        runtime_capabilities=runtime_capabilities,
        consistency_report=consistency_report.to_dict(),
    )
