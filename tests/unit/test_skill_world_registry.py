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

    contabilidad = reg.get_skill("contabilidad_automatica")
    assert contabilidad is not None
    assert contabilidad.world == "contabilidad_automatica"
    assert "reconcile_credit" in contabilidad.intents
    assert "reconcile_credit_accounting" in contabilidad.tools

    semantic = reg.get_skill("semantic_researcher")
    assert semantic is not None
    assert semantic.world == "semantic_researcher"
    assert "semantic_lookup" in semantic.intents
