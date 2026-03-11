from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_build(relative_skill_path: str):
    skill_py = Path("agnostic_agent/skills") / relative_skill_path / "skill.py"
    module_name = f"test_{relative_skill_path}_skill"
    spec = spec_from_file_location(module_name, skill_py)
    module = module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.build


def test_costo_saneamiento_contrato_runs_reconciliation_traceability():
    build = _load_build("costo_saneamiento_contrato")
    skill = build()
    out = skill.run({"user_request": "Explica la trazabilidad del costo para LOC-0004", "credito_id": "LOC-0004"})
    assert out["status"] == "success"
    assert out["outputs"]["intent"] == "contract_traceability"


def test_gobierno_cuentas_contables_generates_entry_proposal():
    build = _load_build("gobierno_cuentas_contables")
    skill = build()
    out = skill.run({"user_request": "Genera un asiento contable para LOC-0004", "credito_id": "LOC-0004"})
    assert out["status"] == "success"
    assert out["outputs"]["intent"] == "generate_accounting_entry"


def test_conciliaciones_alertas_detects_variation_query():
    build = _load_build("conciliaciones_alertas")
    skill = build()
    out = skill.run({"user_request": "Detecta variaciones atipicas en contabilidad"})
    assert out["status"] == "success"
    assert out["outputs"]["intent"] == "detect_significant_variations"


def test_analisis_saneamiento_stage_breakdown_runs():
    build = _load_build("analisis_saneamiento_stage")
    skill = build()
    out = skill.run({"user_request": "Despliega la cartera por estatus y saldo"})
    assert out["status"] == "success"
    assert out["outputs"]["intent"] == "portfolio_breakdown"
