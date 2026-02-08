
from __future__ import annotations

import os
import yaml
import glob
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class Skill:
    name: str
    description: str
    instructions: str
    tools: List[str] = field(default_factory=list)
    knowledge: List[str] = field(default_factory=list)
    
    # Metadata for UI / debugging
    file_path: Optional[str] = None
    enabled: bool = True

class SkillRegistry:
    def __init__(self, skills_dir: str):
        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self.load_skills()

    def load_skills(self):
        """Scans the skills directory for .md files and loads them."""
        self.skills = {}
        if not os.path.isdir(self.skills_dir):
            return

        pattern = os.path.join(self.skills_dir, "*.md")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Parse frontmatter (YAML) and content (Instructions)
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter_raw = parts[1]
                        instructions = parts[2].strip()
                        
                        meta = yaml.safe_load(frontmatter_raw)
                        name = meta.get("name")
                        if name:
                            # Support 'knowledge' or 'kbs' (legacy)
                            kv = meta.get("knowledge") or meta.get("kbs") or []
                            
                            skill = Skill(
                                name=name,
                                description=meta.get("description", ""),
                                instructions=instructions,
                                tools=meta.get("tools", []),
                                knowledge=kv,
                                file_path=file_path
                            )
                            self.skills[name] = skill
            except Exception as e:
                print(f"Error loading skill from {file_path}: {e}")

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        if enabled_only:
             return [s for s in self.skills.values() if s.enabled]
        return list(self.skills.values())

    def set_enabled(self, name: str, enabled: bool):
        if name in self.skills:
            self.skills[name].enabled = enabled
