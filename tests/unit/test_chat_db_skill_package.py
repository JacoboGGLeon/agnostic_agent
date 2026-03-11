from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_chat_db_builder():
    path = Path("agnostic_agent/skills/chat_db/skill.py")
    spec = spec_from_file_location("chat_db_skill_test_module", path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build


def test_chat_db_skill_runs_nl2sql_path():
    build = _load_chat_db_builder()
    skill = build()
    out = skill.run(
        {
            "user_request": "dame informacion del credito LOC-0004",
            "db_path": "contabilidad.db",
            "execute": False,
        }
    )

    assert out["status"] == "success"
    assert out["outputs"]["world"] == "chat_db"
    assert out["outputs"]["intent"] == "query_data"
    assert out["outputs"]["entity_id"] == "LOC-0004"
    assert out["artifacts"][0]["kind"] == "sql_result"


def test_chat_db_skill_runs_schema_path():
    build = _load_chat_db_builder()
    skill = build()
    out = skill.run({"user_request": "muestrame el schema de transacciones.db"})

    assert out["status"] == "success"
    assert out["outputs"]["intent"] == "explain_schema"
    assert out["artifacts"][0]["kind"] == "schema_result"
