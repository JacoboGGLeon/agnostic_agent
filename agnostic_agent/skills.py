from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from agnostic_agent.protocols.smp import validate_skill_manifest


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

        kv = meta.get("knowledge") or meta.get("kbs") or []
        return Skill(
            name=name,
            description=meta.get("description", ""),
            instructions=instructions,
            tools=meta.get("tools", []),
            knowledge=kv,
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

        return Skill(
            name=name.strip(),
            description=str(data.get("description", "")),
            instructions=instructions,
            tools=tool_declared,
            knowledge=knowledge_bindings,
            file_path=str(manifest_path),
            version=str(data.get("version", "")).strip() or None,
            source_type="manifest",
            entrypoint=str(data.get("entrypoint", "")).strip() or None,
            input_schema=str(data.get("input_schema", "")).strip() or None,
            output_schema=str(data.get("output_schema", "")).strip() or None,
            metadata={"manifest": data},
        )

    def load_skills(self) -> None:
        """Scans for markdown and manifest skills and loads them."""
        self.skills = {}
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

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        if enabled_only:
            return [s for s in self.skills.values() if s.enabled]
        return list(self.skills.values())

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self.skills:
            self.skills[name].enabled = enabled
