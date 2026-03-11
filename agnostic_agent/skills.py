from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from agnostic_agent.protocols.smp import validate_skill_manifest
from agnostic_agent.runtime import append_tep_report, assess_skill_maturity


def _parse_semver(value: Optional[str]) -> tuple[int, int, int]:
    if not value:
        return (0, 0, 0)
    parts = str(value).strip().split(".")
    nums: List[int] = []
    for p in parts[:3]:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]


@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: List[str] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)
    world: Optional[str] = None
    intents: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    planner_policy: Dict[str, Any] = field(default_factory=dict)
    summarizer_policy: Dict[str, Any] = field(default_factory=dict)
    validator_policy: Dict[str, Any] = field(default_factory=dict)
    ui: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    hidden: bool = False

    # Metadata for UI / debugging / compatibility
    file_path: Optional[str] = None
    enabled: bool = True
    version: Optional[str] = None
    source_type: str = "markdown"
    entrypoint: Optional[str] = None
    input_schema: Optional[str] = None
    output_schema: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillRegistry:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self.aliases: Dict[str, str] = {}
        self.load_skills()

    def _merge_skill(self, candidate: Skill) -> None:
        existing = self.skills.get(candidate.name)
        if existing is None:
            self.skills[candidate.name] = candidate
            return

        old_ver = _parse_semver(existing.version)
        new_ver = _parse_semver(candidate.version)
        if new_ver > old_ver:
            self.skills[candidate.name] = candidate
            return

        merged_aliases = sorted({*existing.aliases, *candidate.aliases})
        existing.aliases = merged_aliases
        existing.hidden = existing.hidden and candidate.hidden

    def _load_markdown_skill(self, file_path: str) -> Optional[Skill]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.startswith("---"):
            return None
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        frontmatter_raw = parts[1]
        instructions = parts[2].strip()
        meta = yaml.safe_load(frontmatter_raw) or {}
        name = meta.get("name")
        if not name:
            return None

        aliases = meta.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        metadata_node = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        replaces = metadata_node.get("replaces")
        if isinstance(replaces, str) and replaces.strip():
            aliases = list(aliases) + [replaces.strip()]
        elif isinstance(replaces, list):
            aliases = list(aliases) + [str(v).strip() for v in replaces if str(v).strip()]
        ui_node = meta.get("ui") if isinstance(meta.get("ui"), dict) else {}
        hidden = bool(ui_node.get("hidden")) or bool(metadata_node.get("deprecated"))
        kv = meta.get("knowledge") or meta.get("kbs") or []
        return Skill(
            name=name,
            description=meta.get("description", ""),
            instructions=instructions,
            tools=meta.get("tools", []),
            knowledge=kv,
            world=str(meta.get("world", "")).strip() or name,
            intents=[str(i).strip() for i in (meta.get("intents") or []) if str(i).strip()],
            entities=[str(i).strip() for i in (meta.get("entities") or []) if str(i).strip()],
            planner_policy=(meta.get("planner") if isinstance(meta.get("planner"), dict) else {}),
            summarizer_policy=(meta.get("summarizer") if isinstance(meta.get("summarizer"), dict) else {}),
            validator_policy=(meta.get("validator") if isinstance(meta.get("validator"), dict) else {}),
            ui=(meta.get("ui") if isinstance(meta.get("ui"), dict) else {}),
            aliases=[str(a).strip() for a in aliases if str(a).strip()],
            hidden=hidden,
            file_path=file_path,
            version=meta.get("version"),
            source_type="markdown",
            metadata={"frontmatter": meta},
        )

    def _load_manifest_skill(self, manifest_path: Path) -> Optional[Skill]:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return None
        is_valid, _errors = validate_skill_manifest(data, base_path=manifest_path.parent)
        if not is_valid:
            return None

        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return None

        base = manifest_path.parent
        instructions_rel = str(data.get("instructions") or "instructions.md")
        instructions_path = base / instructions_rel
        instructions = ""
        if instructions_path.exists():
            instructions = instructions_path.read_text(encoding="utf-8").strip()

        tool_declared = []
        tools_node = data.get("tools") or {}
        if isinstance(tools_node, dict):
            declared = tools_node.get("declared") or []
            if isinstance(declared, list):
                tool_declared = [str(t).strip() for t in declared if str(t).strip()]

        knowledge_bindings = []
        knowledge_node = data.get("knowledge") or {}
        if isinstance(knowledge_node, dict):
            bindings = knowledge_node.get("bindings") or []
            if isinstance(bindings, list):
                knowledge_bindings = [str(k).strip() for k in bindings if str(k).strip()]

        intents = data.get("intents") or []
        if not isinstance(intents, list):
            intents = []

        entities = data.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        aliases = data.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        metadata_node = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        replaces = metadata_node.get("replaces")
        if isinstance(replaces, str) and replaces.strip():
            aliases = list(aliases) + [replaces.strip()]
        elif isinstance(replaces, list):
            aliases = list(aliases) + [str(v).strip() for v in replaces if str(v).strip()]
        ui_node = data.get("ui") if isinstance(data.get("ui"), dict) else {}
        hidden = bool(ui_node.get("hidden")) or bool(metadata_node.get("deprecated"))

        return Skill(
            name=name.strip(),
            description=str(data.get("description", "")),
            instructions=instructions,
            tools=tool_declared,
            knowledge=knowledge_bindings,
            world=str(data.get("world", "")).strip() or name.strip(),
            intents=[str(i).strip() for i in intents if str(i).strip()],
            entities=[str(i).strip() for i in entities if str(i).strip()],
            planner_policy=(data.get("planner") if isinstance(data.get("planner"), dict) else {}),
            summarizer_policy=(data.get("summarizer") if isinstance(data.get("summarizer"), dict) else {}),
            validator_policy=(data.get("validator") if isinstance(data.get("validator"), dict) else {}),
            ui=(data.get("ui") if isinstance(data.get("ui"), dict) else {}),
            aliases=[str(a).strip() for a in aliases if str(a).strip()],
            hidden=hidden,
            file_path=str(manifest_path),
            version=str(data.get("version", "")).strip() or None,
            source_type="manifest",
            entrypoint=str(data.get("entrypoint", "")).strip() or None,
            input_schema=str(data.get("input_schema", "")).strip() or None,
            output_schema=str(data.get("output_schema", "")).strip() or None,
            metadata={"manifest": data},
        )

    def _certify_loaded_skill(self, skill: Skill) -> None:
        manifest_valid = skill.source_type == "manifest"
        smoke_ok = bool((skill.instructions or "").strip())
        schema_valid = bool(skill.input_schema and skill.output_schema) if skill.source_type == "manifest" else True
        checks = {
            "manifest_valid": manifest_valid,
            "smoke_ok": smoke_ok,
            "schema_valid": schema_valid,
            "errors_normalized": True,
            "tool_contracts": bool(skill.tools),
            "knowledge_contracts": bool(skill.knowledge),
            "artifacts_emitted": False,
            "observability_complete": False,
            "version_stable": bool(skill.version),
        }
        report = assess_skill_maturity(
            skill_name=skill.name,
            checks=checks,
            notes={"source_type": skill.source_type},
        )
        skill.metadata["certification"] = report.model_dump()
        skill.metadata["maturity_level"] = report.level

        auto_record = os.getenv("AGNOSTIC_TEP_AUTO_RECORD", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if auto_record:
            report_path = os.getenv("AGNOSTIC_TEP_REPORT_PATH", "documents/tep_reports.json")
            append_tep_report(report_path, report)

    def load_skills(self) -> None:
        """Scans for markdown and manifest skills and loads them."""
        self.skills = {}
        self.aliases = {}
        if not os.path.isdir(self.skills_dir):
            return

        # 1) Existing markdown skill format.
        pattern = os.path.join(self.skills_dir, "*.md")
        for file_path in glob.glob(pattern):
            try:
                skill = self._load_markdown_skill(file_path)
                if skill:
                    self._merge_skill(skill)
            except Exception as e:
                print(f"Error loading markdown skill from {file_path}: {e}")

        # 2) Manifest package format.
        root = Path(self.skills_dir)
        for manifest_path in root.rglob("manifest.yaml"):
            try:
                skill = self._load_manifest_skill(manifest_path)
                if skill:
                    self._merge_skill(skill)
            except Exception as e:
                print(f"Error loading manifest skill from {manifest_path}: {e}")

        for skill in self.skills.values():
            self._certify_loaded_skill(skill)
            for alias in skill.aliases:
                if alias and alias != skill.name:
                    self.aliases[alias] = skill.name

    def get_skill(self, name: str) -> Optional[Skill]:
        skill = self.skills.get(name)
        if skill is not None and not skill.hidden:
            return skill
        canonical = self.aliases.get(name)
        if canonical:
            return self.skills.get(canonical)
        return skill

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        visible_skills = [s for s in self.skills.values() if not s.hidden]
        if enabled_only:
            return [s for s in visible_skills if s.enabled]
        return visible_skills

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self.skills:
            self.skills[name].enabled = enabled

    def get_world(self, name: str) -> Optional[Skill]:
        skill = self.get_skill(name)
        if skill is not None:
            return skill
        for candidate in self.skills.values():
            if candidate.hidden:
                continue
            if (candidate.world or candidate.name) == name:
                return candidate
        return None
