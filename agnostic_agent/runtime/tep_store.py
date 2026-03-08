from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agnostic_agent.runtime.certification import CertificationReport


def load_tep_reports(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def append_tep_report(path: str | Path, report: CertificationReport) -> None:
    p = Path(path)
    existing = load_tep_reports(p)
    existing.append(report.model_dump())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
