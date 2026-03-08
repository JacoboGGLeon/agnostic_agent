from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_MANIFEST_FIELDS = (
    "api_version",
    "kind",
    "name",
    "version",
    "entrypoint",
    "instructions",
    "input_schema",
    "output_schema",
)


def validate_skill_manifest(
    manifest: Dict[str, Any],
    *,
    base_path: str | Path | None = None,
) -> Tuple[bool, List[str]]:
    """
    Minimal SMP validator.
    Returns (is_valid, errors) with deterministic ordering.
    """
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return False, ["manifest must be a dictionary"]

    for field in REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(field)
        if value in (None, ""):
            errors.append(f"missing required field: {field}")

    if manifest.get("api_version") not in {"skill/v1"}:
        errors.append("unsupported api_version")
    if manifest.get("kind") not in {"skill"}:
        errors.append("unsupported kind")

    if base_path is not None:
        root = Path(base_path)
        for field in ("instructions", "input_schema", "output_schema"):
            rel = manifest.get(field)
            if rel in (None, ""):
                continue
            if not (root / str(rel)).exists():
                errors.append(f"missing file for {field}: {rel}")

    return len(errors) == 0, errors
