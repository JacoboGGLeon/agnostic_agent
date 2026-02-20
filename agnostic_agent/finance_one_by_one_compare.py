from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

from agnostic_agent.agent import Agent


def _default_finance_dir() -> Path:
    # Repo layout expected:
    # AVANTECK.TEAM/
    #   - agnostic_agent/
    #   - ais/examples/finance/
    workspace_root = Path(__file__).resolve().parents[2]
    return workspace_root / "ais" / "examples" / "finance"


def _read_batch_candidates(limit: int) -> List[Tuple[str, str, float]]:
    finance_dir = _default_finance_dir()
    accounting_db = Path(os.getenv("AGNOSTIC_FIN_ACC_DB", str(finance_dir / "contabilidad.db")))
    if not accounting_db.exists():
        raise FileNotFoundError(f"No se encontro base de datos: {accounting_db}")

    conn = sqlite3.connect(str(accounting_db))
    cur = conn.cursor()
    cur.execute(
        """
        SELECT credito_id, estatus, saldo_total
        FROM estados_cuenta
        ORDER BY credito_id ASC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = cur.fetchall()
    conn.close()
    return [(str(cid), str(estatus), float(saldo)) for cid, estatus, saldo in rows]


def main() -> None:
    limit = int(os.getenv("FINANCE_COMPARE_LIMIT", "12"))
    rows = _read_batch_candidates(limit)

    agent = Agent.init()

    print(f"Comparacion 1-a-1 sobre {len(rows)} creditos")
    print("=" * 80)
    for credito_id, estatus, saldo_total in rows:
        prompt = (
            f"Realiza la conciliacion del credito {credito_id}. "
            f"Estatus: {estatus}. Saldo: {saldo_total:.2f}."
        )
        out = agent.run_turn(
            {
                "user_prompt": prompt,
                "session_id": f"finance-{credito_id}",
                "metadata": {
                    "skills_allowlist": ["contabilidad_instantanea"],
                },
            }
        )
        answer = (
            out.get("user_out", {})
            .get("final_answer", "")
            .strip()
            .replace("\n", " ")
        )
        print(f"[{credito_id}] {answer}")


if __name__ == "__main__":
    main()
