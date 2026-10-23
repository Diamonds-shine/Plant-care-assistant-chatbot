"""Load and validate the plant-care knowledge base.

The knowledge base is two JSON files:
  * plant_care_core.json       - hand-written answers to general questions
  * plant_care_generated.json  - built from the FloraDB houseplant dataset
                                 by scripts/build_knowledge_base.py

Both share one schema. Validation runs at start-up so a broken file stops
the server with a clear message instead of producing odd answers later.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = ("id", "topic", "question", "alternatives", "keywords", "answer", "source")

TOPIC_LABELS = {
    "watering": "Watering",
    "leaf_problems": "Leaf Problems",
    "light_environment": "Light & Environment",
    "soil_pots": "Soil & Pots",
    "fertilizer": "Fertilizer & Nutrients",
    "pests_maintenance": "Pests & Maintenance",
    "general_care": "General Care",
    "plant_profiles": "Plant Profiles",
    "pet_safety": "Pet Safety",
    "plant_recommendations": "Plant Recommendations",
}


class KnowledgeBaseError(Exception):
    """Raised when a knowledge-base file is missing, malformed or invalid."""


def _validate_entry(entry: object, index: int, source_name: str) -> None:
    where = f"{source_name}, entry #{index}"
    if not isinstance(entry, dict):
        raise KnowledgeBaseError(f"{where}: each entry must be a JSON object.")
    missing = [name for name in REQUIRED_FIELDS if name not in entry]
    if missing:
        raise KnowledgeBaseError(f"{where} (id={entry.get('id')!r}): missing field(s) {missing}.")
    for name in ("id", "topic", "question", "answer", "source"):
        if not isinstance(entry[name], str) or not entry[name].strip():
            raise KnowledgeBaseError(f"{where} (id={entry.get('id')!r}): '{name}' must be a non-empty string.")
    for name in ("alternatives", "keywords"):
        values = entry[name]
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
            raise KnowledgeBaseError(f"{where} (id={entry['id']!r}): '{name}' must be a list of non-empty strings.")
    if entry["topic"] not in TOPIC_LABELS:
        raise KnowledgeBaseError(f"{where} (id={entry['id']!r}): unknown topic {entry['topic']!r}.")


def load_entries(path: Path) -> list[dict]:
    """Read and validate one knowledge-base JSON file."""
    path = Path(path)
    if not path.exists():
        raise KnowledgeBaseError(f"Knowledge-base file not found: {path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"{path.name} is not valid JSON (line {exc.lineno}): {exc.msg}") from exc
    if not isinstance(data, list):
        raise KnowledgeBaseError(f"{path.name} must contain a JSON list of entries.")
    if not data:
        raise KnowledgeBaseError(f"{path.name} is empty; add at least one entry.")
    for index, entry in enumerate(data, start=1):
        _validate_entry(entry, index, path.name)
    return data


def load_knowledge_base(*paths: Path) -> list[dict]:
    """Load several files (core first), checking that ids are unique across all of them."""
    entries: list[dict] = []
    seen: dict[str, str] = {}
    for path in paths:
        for entry in load_entries(path):
            if entry["id"] in seen:
                raise KnowledgeBaseError(
                    f"Duplicate id {entry['id']!r} in {Path(path).name} (already defined in {seen[entry['id']]})."
                )
            seen[entry["id"]] = Path(path).name
            entries.append(entry)
    return entries
