import importlib.util
from pathlib import Path

from agnostic_agent.skills import SkillRegistry


def _load_skill_module():
    skill_path = Path("agnostic_agent/skills/contabilidad_instantanea/skill.py")
    spec = importlib.util.spec_from_file_location("contabilidad_instantanea_skill", skill_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contabilidad_instantanea_skill_package_is_discoverable():
    repo_skills_dir = Path("agnostic_agent/skills")
    reg = SkillRegistry(str(repo_skills_dir))
    skill = reg.get_skill("contabilidad_instantanea")
    assert skill is not None
    assert skill.source_type == "manifest"
    assert skill.version == "1.1.0"
    assert skill.tools == ["query_transactions_db", "query_accounting_db"]
    assert skill.input_schema == "schemas/input.schema.json"
    assert skill.output_schema == "schemas/output.schema.json"


def test_contabilidad_instantanea_build_and_run_contract():
    module = _load_skill_module()
    instance = module.build()

    missing = instance.run({"credito_id": "LOC-0004"})
    assert missing["status"] == "error"
    assert missing["outputs"]["ok"] is False
    assert "estatus" in missing["outputs"]["missing_fields"]
    assert "saldo_total" in missing["outputs"]["missing_fields"]

    ok = instance.run({"credito_id": "LOC-0004", "estatus": "Desembolsado", "saldo_total": 29440.64})
    assert ok["status"] == "success"
    out = ok["outputs"]
    assert out["ok"] is True
    assert out["estado_conciliacion"] == "PENDIENTE_EJECUCION_TOOLS"
    calls = out["planned_tool_calls"]
    assert len(calls) == 2
    assert calls[0]["tool"] == "query_transactions_db"
    assert calls[1]["tool"] == "query_accounting_db"
    assert ok["metrics"]["planned_calls"] == 2
