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

    contabilidad = reg.get_skill("contabilidad_automatica")
    assert contabilidad is not None
    assert contabilidad.world == "contabilidad_automatica"
    assert "reconcile_credit" in contabilidad.intents
    assert "reconcile_credit_accounting" in contabilidad.tools
    assert "lookup_finance_rule" in contabilidad.tools
    assert contabilidad.intent_entity_requirements["explain_rule"]["required"] == ["estatus"]

    semantic = reg.get_skill("semantic_researcher")
    assert semantic is not None
    assert semantic.world == "semantic_researcher"
    assert "semantic_lookup" in semantic.intents
    assert semantic.consistency_report.get("status") in {"healthy", "degraded", "broken"}

    visible_names = sorted(skill.name for skill in reg.list_skills(enabled_only=False))
    assert visible_names == ["chat_db", "contabilidad_automatica", "semantic_researcher"]


def test_legacy_skill_aliases_resolve_to_canonical_worlds():
    repo_skills_dir = Path("agnostic_agent/skills")
    reg = SkillRegistry(str(repo_skills_dir))

    assert reg.get_skill("contabilidad_instantanea").name == "contabilidad_automatica"
    assert reg.get_skill("nl2sql_sqlite").name == "chat_db"
