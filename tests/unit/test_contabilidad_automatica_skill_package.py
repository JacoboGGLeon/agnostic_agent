import importlib.util
import sys
from pathlib import Path


def _load_skill_module():
    skill_path = Path("agnostic_agent/skills/contabilidad_automatica/skill.py")
    spec = importlib.util.spec_from_file_location("contabilidad_automatica_skill", skill_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_contabilidad_automatica_build_and_query_runtime():
    module = _load_skill_module()
    instance = module.build()

    missing = instance.run({})
    assert missing["status"] == "error"

    ok = instance.run({"user_request": "dame el saldo total del credito LOC-0004", "execute": False})
    assert ok["status"] == "success"
    out = ok["outputs"]
    assert out["world"] == "contabilidad_automatica"
    assert out["intent"] == "query_financial_data"
    assert out["credito_id"] == "LOC-0004"
    assert out["db_path"].endswith("contabilidad.db")
    assert ok["artifacts"][0]["kind"] == "query_result"


def test_contabilidad_automatica_build_and_rule_runtime():
    module = _load_skill_module()
    instance = module.build()

    ok = instance.run({"user_request": "dime la tasa de saneamiento para estatus Vigente / Al corriente"})
    assert ok["status"] == "success"
    assert ok["outputs"]["intent"] == "explain_rule"
    assert len(ok["artifacts"]) >= 2
