from pathlib import Path

from agnostic_agent.skills import SkillRegistry


def test_skill_worlds_are_discoverable():
    repo_skills_dir = Path("agnostic_agent/skills")
    reg = SkillRegistry(str(repo_skills_dir))

    chat_db = reg.get_skill("chat_db")
    assert chat_db is not None
    assert chat_db.world == "chat_db"
    assert "nl2sql" in chat_db.tools
    assert "query_data" in chat_db.intents
    assert isinstance(chat_db.consistency_report, dict)
    assert isinstance(chat_db.capability_contract, dict)
    assert chat_db.consistency_report.get("status") == "healthy"

    contabilidad = reg.get_skill("contabilidad_automatica")
    assert contabilidad is not None
    assert contabilidad.world == "contabilidad_automatica"
    assert "reconcile_credit" in contabilidad.intents
    assert "reconcile_credit_accounting" in contabilidad.tools
    assert "lookup_finance_rule" in contabilidad.tools
    assert contabilidad.intent_entity_requirements["explain_rule"]["required"] == ["estatus"]
    assert contabilidad.consistency_report.get("status") == "healthy"

    semantic = reg.get_skill("semantic_researcher")
    assert semantic is not None
    assert semantic.world == "semantic_researcher"
    assert "semantic_lookup" in semantic.intents
    assert semantic.consistency_report.get("status") == "healthy"

    costo = reg.get_skill("costo_saneamiento_contrato")
    assert costo is not None
    assert costo.world == "costo_saneamiento_contrato"
    assert "contract_traceability" in costo.intents
    assert "reconcile_credit_accounting" in costo.tools
    assert costo.consistency_report.get("status") == "healthy"

    gobierno = reg.get_skill("gobierno_cuentas_contables")
    assert gobierno is not None
    assert gobierno.world == "gobierno_cuentas_contables"
    assert "generate_accounting_entry" in gobierno.intents
    assert "finance_sources_status" in gobierno.tools
    assert gobierno.consistency_report.get("status") == "healthy"

    alertas = reg.get_skill("conciliaciones_alertas")
    assert alertas is not None
    assert alertas.world == "conciliaciones_alertas"
    assert "detect_significant_variations" in alertas.intents
    assert "finance_sources_status" in alertas.tools
    assert alertas.consistency_report.get("status") == "healthy"

    analisis = reg.get_skill("analisis_saneamiento_stage")
    assert analisis is not None
    assert analisis.world == "analisis_saneamiento_stage"
    assert "generate_trends_and_projections" in analisis.intents
    assert "lookup_finance_dictionary" in analisis.tools
    assert analisis.consistency_report.get("status") == "healthy"

    visible_names = sorted(skill.name for skill in reg.list_skills(enabled_only=False))
    assert visible_names == [
        "analisis_saneamiento_stage",
        "chat_db",
        "conciliaciones_alertas",
        "contabilidad_automatica",
        "costo_saneamiento_contrato",
        "gobierno_cuentas_contables",
        "semantic_researcher",
    ]


def test_legacy_skill_aliases_resolve_to_canonical_worlds():
    repo_skills_dir = Path("agnostic_agent/skills")
    reg = SkillRegistry(str(repo_skills_dir))

    assert reg.get_skill("contabilidad_instantanea").name == "contabilidad_automatica"
    assert reg.get_skill("nl2sql_sqlite").name == "chat_db"
