from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from agnostic_agent.protocols.smp import validate_skill_manifest
from agnostic_agent.skill_contracts import SkillConsistencyReport


def _existing_paths(skill: Any) -> tuple[int, int]:
    declared = list(skill.knowledge or [])
    if not declared or not skill.file_path:
        return 0, len(declared)
    manifest_path = Path(skill.file_path)
    base = manifest_path.parent
    resolved = 0
    for rel_path in declared:
        if (base / str(rel_path)).exists():
            resolved += 1
    return resolved, len(declared)


def _planner_policy_issues(skill: Any, tool_registry: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    policy = skill.planner_policy if isinstance(skill.planner_policy, dict) else {}
    intent_to_tools = policy.get("intent_to_tools") if isinstance(policy.get("intent_to_tools"), dict) else {}
    if not skill.intents:
        return issues
    for intent in skill.intents:
        mapped_tools = intent_to_tools.get(intent)
        if not isinstance(mapped_tools, list) or not mapped_tools:
            issues.append(f"intent `{intent}` has no planner.intent_to_tools mapping")
            continue
        for tool_name in mapped_tools:
            if tool_name not in skill.tools:
                issues.append(f"intent `{intent}` references undeclared tool `{tool_name}`")
            if tool_name not in tool_registry:
                issues.append(f"intent `{intent}` references missing tool `{tool_name}`")
    return issues


def _instruction_issues(skill: Any) -> List[str]:
    issues: List[str] = []
    instruction_text = str(skill.instructions or "").lower()
    normalized_instruction_text = "".join(
        ch for ch in unicodedata.normalize("NFKD", instruction_text) if not unicodedata.combining(ch)
    )
    if not instruction_text.strip():
        return ["instructions.md is empty"]
    world_name = str(skill.world or skill.name).replace("_", " ").lower()
    if world_name not in instruction_text and str(skill.name).replace("_", " ").lower() not in instruction_text:
        issues.append("instructions do not mention the skill world or canonical skill name")

    expectations = skill.instruction_expectations if isinstance(getattr(skill, "instruction_expectations", {}), dict) else {}
    policy = skill.planner_policy if isinstance(skill.planner_policy, dict) else {}
    intent_to_tools = policy.get("intent_to_tools") if isinstance(policy.get("intent_to_tools"), dict) else {}
    critical_tools: List[str] = []
    expected_tools = expectations.get("critical_tools") if isinstance(expectations.get("critical_tools"), list) else []
    for tool_name in expected_tools:
        normalized = str(tool_name).strip()
        if normalized and normalized not in critical_tools:
            critical_tools.append(normalized)
    if not critical_tools:
        for mapped_tools in intent_to_tools.values():
            if isinstance(mapped_tools, list):
                for tool_name in mapped_tools:
                    if tool_name not in critical_tools:
                        critical_tools.append(tool_name)
    missing_tools = [tool_name for tool_name in critical_tools if tool_name.lower() not in instruction_text]
    if missing_tools:
        issues.append(f"instructions omit critical tools: {', '.join(missing_tools)}")

    def _topic_is_mentioned(topic: str) -> bool:
        raw = str(topic or "").strip().lower()
        if not raw:
            return True
        humanized = raw.replace("_", " ")
        kebab = raw.replace("_", "-")
        normalized_candidates = {
            raw,
            humanized,
            kebab,
        }
        normalized_candidates |= {
            "".join(ch for ch in unicodedata.normalize("NFKD", candidate) if not unicodedata.combining(ch))
            for candidate in list(normalized_candidates)
        }
        return any(candidate in normalized_instruction_text for candidate in normalized_candidates if candidate)

    required_topics = expectations.get("required_topics") if isinstance(expectations.get("required_topics"), list) else []
    topics = [str(topic).strip() for topic in required_topics if str(topic).strip()] or list(skill.intents or [])
    missing_intents = [intent for intent in topics if not _topic_is_mentioned(intent)]
    if len(missing_intents) == len(topics) and missing_intents:
        issues.append("instructions do not describe the declared intents")
    return issues


def _runtime_alignment_issues(skill: Any) -> List[str]:
    issues: List[str] = []
    policy = skill.planner_policy if isinstance(skill.planner_policy, dict) else {}
    intent_to_tools = policy.get("intent_to_tools") if isinstance(policy.get("intent_to_tools"), dict) else {}
    planner_intents = set(intent_to_tools.keys())
    runtime_intents = skill.metadata.get("runtime_intents") if isinstance(skill.metadata, dict) else None
    skill_runtime_intents = set(runtime_intents or [])
    declared_intents = set(skill.intents or [])
    if declared_intents and not planner_intents.issuperset(declared_intents):
        missing = sorted(declared_intents - planner_intents)
        issues.append(f"planner runtime does not cover declared intents: {', '.join(missing)}")
    if skill.source_type == "manifest" and not skill_runtime_intents.issuperset(declared_intents):
        missing = sorted(declared_intents - skill_runtime_intents)
        issues.append(f"skill runtime does not mention declared intents: {', '.join(missing)}")
    return issues


def build_skill_consistency_report(skill: Any, tool_registry: Dict[str, Any]) -> SkillConsistencyReport:
    manifest_data = skill.metadata.get("manifest") if isinstance(skill.metadata, dict) else None
    manifest_valid = False
    manifest_errors: List[str] = []
    if isinstance(manifest_data, dict) and skill.file_path:
        manifest_valid, manifest_errors = validate_skill_manifest(manifest_data, base_path=Path(skill.file_path).parent)

    tools_declared = len(skill.tools or [])
    tools_resolved = sum(1 for tool_name in (skill.tools or []) if tool_name in tool_registry)
    knowledge_resolved, knowledge_declared = _existing_paths(skill)
    planner_issues = _planner_policy_issues(skill, tool_registry)
    instruction_issues = _instruction_issues(skill)
    runtime_issues = _runtime_alignment_issues(skill)

    issues = list(manifest_errors) + planner_issues + runtime_issues
    warnings = instruction_issues[:]
    planner_policy_valid = len(planner_issues) == 0
    runtime_alignment_valid = len(runtime_issues) == 0
    instructions_valid = len(instruction_issues) == 0

    if not manifest_valid or tools_resolved != tools_declared or knowledge_resolved != knowledge_declared:
        status = "broken"
    elif issues or warnings:
        status = "degraded"
    else:
        status = "healthy"

    return SkillConsistencyReport(
        manifest_valid=manifest_valid,
        instructions_valid=instructions_valid,
        tools_resolved=tools_resolved,
        tools_declared=tools_declared,
        knowledge_resolved=knowledge_resolved,
        knowledge_declared=knowledge_declared,
        planner_policy_valid=planner_policy_valid,
        runtime_alignment_valid=runtime_alignment_valid,
        issues=issues,
        warnings=warnings,
        status=status,
    )
