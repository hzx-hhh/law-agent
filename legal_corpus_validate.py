from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def validate(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    metadata = data.get("metadata", {})
    for key in ("version", "source_url", "effective_from"):
        if not metadata.get(key):
            errors.append(f"metadata.{key} missing")
    for item in data.get("laws", []):
        for key in ("source_url", "effective_from"):
            if not item.get(key):
                errors.append(f"{item.get('id')}.{key} missing")
        try:
            date.fromisoformat(item["effective_from"])
        except (KeyError, ValueError):
            errors.append(f"{item.get('id')}.effective_from invalid")
    return errors


if __name__ == "__main__":
    errors = validate(Path(__file__).parent / "laws_db.json")
    print("CORPUS_VALID" if not errors else "\n".join(errors))
    raise SystemExit(0 if not errors else 1)
