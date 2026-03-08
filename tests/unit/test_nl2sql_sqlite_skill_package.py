from pathlib import Path

from agnostic_agent.skills import SkillRegistry


def test_nl2sql_sqlite_skill_package_is_discoverable():
    repo_skills_dir = Path("agnostic_agent/skills")
    reg = SkillRegistry(str(repo_skills_dir))
    skill = reg.get_skill("nl2sql_sqlite")
    assert skill is not None
    assert skill.source_type == "manifest"
    assert "nl2sql_sqlite" in skill.tools
    assert skill.input_schema == "schemas/input.schema.json"
    assert skill.output_schema == "schemas/output.schema.json"
