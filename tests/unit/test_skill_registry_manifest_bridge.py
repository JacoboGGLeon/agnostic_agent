from pathlib import Path

from agnostic_agent.skills import SkillRegistry


def test_skill_registry_loads_markdown_and_manifest(tmp_path: Path):
    md_skill = tmp_path / "text_basic.md"
    md_skill.write_text(
        """---
name: text_basic
description: markdown skill
tools: [to_upper]
knowledge: []
version: 0.1.0
---
markdown instructions
""",
        encoding="utf-8",
    )

    pkg = tmp_path / "pkg_skill"
    schemas = pkg / "schemas"
    schemas.mkdir(parents=True)
    (schemas / "input.schema.json").write_text("{}", encoding="utf-8")
    (schemas / "output.schema.json").write_text("{}", encoding="utf-8")
    (pkg / "instructions.md").write_text("manifest instructions", encoding="utf-8")
    (pkg / "skill.py").write_text("def build():\n    return None\n", encoding="utf-8")
    (pkg / "manifest.yaml").write_text(
        """api_version: skill/v1
kind: skill
name: semantic_researcher
version: 0.2.0
description: manifest skill
entrypoint: skill:build
instructions: instructions.md
input_schema: schemas/input.schema.json
output_schema: schemas/output.schema.json
tools:
  declared: [search_chunks]
knowledge:
  bindings: [docs]
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(str(tmp_path))
    names = {s.name for s in reg.list_skills()}
    assert "text_basic" in names
    assert "semantic_researcher" in names
    assert reg.get_skill("semantic_researcher").source_type == "manifest"


def test_manifest_version_precedence_over_markdown(tmp_path: Path):
    md_skill = tmp_path / "router.md"
    md_skill.write_text(
        """---
name: router
description: old markdown
tools: [basic_tool]
knowledge: []
version: 0.1.0
---
old instructions
""",
        encoding="utf-8",
    )

    pkg = tmp_path / "router_pkg"
    schemas = pkg / "schemas"
    schemas.mkdir(parents=True)
    (schemas / "input.schema.json").write_text("{}", encoding="utf-8")
    (schemas / "output.schema.json").write_text("{}", encoding="utf-8")
    (pkg / "instructions.md").write_text("new instructions", encoding="utf-8")
    (pkg / "skill.py").write_text("def build():\n    return None\n", encoding="utf-8")
    (pkg / "manifest.yaml").write_text(
        """api_version: skill/v1
kind: skill
name: router
version: 0.2.0
entrypoint: skill:build
instructions: instructions.md
input_schema: schemas/input.schema.json
output_schema: schemas/output.schema.json
tools:
  declared: [advanced_tool]
""",
        encoding="utf-8",
    )

    reg = SkillRegistry(str(tmp_path))
    router = reg.get_skill("router")
    assert router is not None
    assert router.source_type == "manifest"
    assert router.version == "0.2.0"
    assert "advanced_tool" in router.tools


def test_invalid_manifest_is_skipped_by_smp_validation(tmp_path: Path):
    bad_pkg = tmp_path / "bad_skill"
    bad_pkg.mkdir(parents=True)
    (bad_pkg / "manifest.yaml").write_text(
        """api_version: skill/v1
kind: skill
name: bad_skill
version: 0.1.0
entrypoint: skill:build
instructions: missing_instructions.md
input_schema: missing_input.json
output_schema: missing_output.json
""",
        encoding="utf-8",
    )
    reg = SkillRegistry(str(tmp_path))
    assert reg.get_skill("bad_skill") is None
