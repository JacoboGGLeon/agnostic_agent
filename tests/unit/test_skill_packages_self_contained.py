from pathlib import Path

import yaml


SKILLS_ROOT = Path("agnostic_agent/skills")


def _manifest_paths():
    return sorted(SKILLS_ROOT.rglob("manifest.yaml"))


def test_all_manifest_skills_are_self_contained():
    manifests = _manifest_paths()
    assert manifests, "No manifest skills found"

    for manifest_path in manifests:
        base = manifest_path.parent
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}

        # Core package files.
        assert (base / "instructions.md").exists(), f"missing instructions for {base.name}"
        assert (base / "skill.py").exists(), f"missing skill.py for {base.name}"
        assert (base / "schemas" / "input.schema.json").exists(), f"missing input schema for {base.name}"
        assert (base / "schemas" / "output.schema.json").exists(), f"missing output schema for {base.name}"

        # Local tool bundle.
        assert (base / "tools").exists(), f"missing tools dir for {base.name}"
        assert (base / "tools" / "contracts.yaml").exists(), f"missing tool contracts for {base.name}"
        assert (base / "tools" / "local_tools.py").exists(), f"missing local tool wrapper for {base.name}"

        # Local knowledge bundle.
        assert (base / "knowledge").exists(), f"missing knowledge dir for {base.name}"

        bindings = ((data.get("knowledge") or {}).get("bindings") or [])
        for item in bindings:
            if isinstance(item, str) and item.startswith("knowledge/"):
                assert (base / item).exists(), f"missing knowledge binding '{item}' for {base.name}"

        declared_tools = ((data.get("tools") or {}).get("declared") or [])
        contracts_text = (base / "tools" / "contracts.yaml").read_text(encoding="utf-8")
        for tool_name in declared_tools:
            assert f"name: {tool_name}" in contracts_text, f"missing contract for tool '{tool_name}' in {base.name}"
