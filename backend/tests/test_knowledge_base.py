import json

import pytest

import config
from knowledge_base import KnowledgeBaseError, load_entries, load_knowledge_base
from scripts.build_knowledge_base import build, load_rows, to_json


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


VALID = {
    "id": "x1", "topic": "watering", "question": "Q?", "alternatives": ["A?"],
    "keywords": ["k"], "answer": "Answer.", "source": "curated",
}


def test_real_knowledge_base_loads():
    entries = load_knowledge_base(config.CORE_KB_PATH, config.GENERATED_KB_PATH)
    assert len([e for e in entries if e["source"] == "curated"]) >= 30
    assert len(entries) > 300


def test_related_ids_point_to_real_entries():
    entries = load_knowledge_base(config.CORE_KB_PATH, config.GENERATED_KB_PATH)
    ids = {e["id"] for e in entries}
    assert all(r in ids for e in entries for r in e.get("related", []))


def test_missing_field_is_reported(tmp_path):
    broken = {k: v for k, v in VALID.items() if k != "answer"}
    with pytest.raises(KnowledgeBaseError, match="answer"):
        load_entries(write(tmp_path, "kb.json", [broken]))


def test_bad_json_is_reported(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="not valid JSON"):
        load_entries(write(tmp_path, "kb.json", "[{oops"))


def test_empty_file_is_reported(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="empty"):
        load_entries(write(tmp_path, "kb.json", []))


def test_duplicate_ids_across_files_are_reported(tmp_path):
    a = write(tmp_path, "a.json", [VALID])
    b = write(tmp_path, "b.json", [VALID])
    with pytest.raises(KnowledgeBaseError, match="Duplicate"):
        load_knowledge_base(a, b)


def test_generated_file_is_up_to_date_with_dataset():
    """Re-running the builder must reproduce the committed JSON exactly."""
    fresh = to_json(build(load_rows(config.RAW_DATASET_PATH)))
    assert fresh == config.GENERATED_KB_PATH.read_text(encoding="utf-8")


def test_builder_respects_data_quality_flags():
    entries = build(load_rows(config.RAW_DATASET_PATH))
    low_confidence = [e for e in entries if e["data_quality"]["care_confidence"] == "low" and e["plant"]]
    assert low_confidence and all(e["topic"] == "pet_safety" for e in low_confidence)
    unknown = [e for e in entries if e["data_quality"]["toxicity_status"] == "unknown" and e["topic"] == "pet_safety"]
    assert unknown and all("not safe to assume" in e["answer"] for e in unknown)


def test_watering_answers_tell_user_to_check_soil():
    entries = build(load_rows(config.RAW_DATASET_PATH))
    watering = [e for e in entries if e["id"].endswith("_water") and e["plant"]]
    assert watering and all("rough guide" in e["answer"] for e in watering)
