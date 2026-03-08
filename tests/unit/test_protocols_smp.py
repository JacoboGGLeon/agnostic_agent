from pathlib import Path

from agnostic_agent.protocols.smp import validate_skill_manifest


def test_validate_skill_manifest_ok(tmp_path: Path):
    (tmp_path / "instructions.md").write_text("x", encoding="utf-8")
    (tmp_path / "in.json").write_text("{}", encoding="utf-8")
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    manifest = {
        "api_version": "skill/v1",
        "kind": "skill",
        "name": "demo",
        "version": "0.1.0",
        "entrypoint": "skill:build",
        "instructions": "instructions.md",
        "input_schema": "in.json",
        "output_schema": "out.json",
    }
    ok, errors = validate_skill_manifest(manifest, base_path=tmp_path)
    assert ok is True
    assert errors == []


def test_validate_skill_manifest_missing_required_fields():
    manifest = {"name": "demo"}
    ok, errors = validate_skill_manifest(manifest)
    assert ok is False
    assert any("missing required field: api_version" in e for e in errors)
