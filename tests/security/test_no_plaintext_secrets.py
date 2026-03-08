from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
YAML_GLOBS = ("*.yaml", "*.yml")

# Common OpenAI-style secret prefixes.
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{20,}\b"),
]


def _iter_yaml_files():
    for pattern in YAML_GLOBS:
        for p in ROOT.rglob(pattern):
            rel = p.relative_to(ROOT)
            parts = rel.parts
            if parts[0].startswith(".venv"):
                continue
            if ".git" in parts:
                continue
            yield p


def test_no_plaintext_api_keys_in_yaml():
    offenders = []
    for path in _iter_yaml_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Allow placeholders like ${OPENAI_API_KEY} or EMPTY.
        if "${OPENAI_API_KEY}" in text:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                offenders.append(str(path.relative_to(ROOT)))
                break
    assert offenders == [], f"Potential plaintext secrets found in YAML: {offenders}"
