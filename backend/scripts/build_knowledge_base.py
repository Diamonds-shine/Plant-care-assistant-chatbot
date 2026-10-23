"""Convert the FloraDB houseplant dataset (CSV) into chatbot Q&A entries.

Run from the backend folder:

    python3 scripts/build_knowledge_base.py

Input : data/raw/floradb_houseplants_sample.csv   (100 plants, one row each)
Output: knowledge_base/plant_care_generated.json  (question/answer entries)

Why this step exists: the dataset is a *table* (light level, watering days,
toxicity flags...), but a retrieval chatbot needs *questions and answers*.
For every plant we fill a few answer templates with that plant's values and
write several phrasings of the matching question. The output is sorted and
written the same way every time, so re-running produces an identical file.

Data-quality rules (from the dataset card):
  * care_confidence "high"/"medium" -> care values are used (medium answers
    say the values are estimated from the plant family).
  * care_confidence "low" -> care values are generic defaults, so only the
    pet-safety entry is generated for that plant.
  * toxicity_status "unknown" is never reported as "safe".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import config  # noqa: E402
from nlp.synonyms import PLANT_ALIASES  # noqa: E402

SOURCE = "floradb"
CARE_CONFIDENCE_USED = {"high", "medium"}

# Explicit, documented corrections applied to the raw data (none needed for
# the current FloraDB sample; kept so any future fix is visible and reviewed).
CORRECTIONS: dict[str, dict[str, str]] = {}

LIGHT_PHRASES = {
    "Bright Indirect": "bright, indirect light: near a window, but out of harsh midday sun",
    "Medium to Bright": "medium to bright indirect light",
    "Medium": "medium, indirect light, such as a spot a few metres from a window",
    "Medium to Low": "medium light, and it tolerates lower light (it will just grow more slowly)",
    "Low": "low light well, although it grows faster in medium indirect light",
    "Low to Bright": "anything from low light to bright indirect light, which makes it very adaptable",
    "Direct Sun": "plenty of direct sun: a sunny window with several hours of sun a day",
}
LOW_LIGHT_LEVELS = {"Low", "Medium to Low", "Low to Bright"}


def an(name: str) -> str:
    """'a Snake Plant' / 'an Aloe Vera'."""
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("'", "")).strip("_")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cleaned = []
    for row in rows:
        row = {key: (value or "").strip() for key, value in row.items()}
        row.update(CORRECTIONS.get(row["common_name"], {}))
        if not row["common_name"] or not row["scientific_name"]:
            print(f"  skipped row {row.get('plant_id')}: missing name", file=sys.stderr)
            continue
        cleaned.append(row)
    return cleaned


def toxicity(row: dict) -> str:
    """Return 'toxic', 'non_toxic', 'likely_toxic' or 'unknown'."""
    status = row["toxicity_status"]
    toxic_flags = {row["is_toxic_to_cats"], row["is_toxic_to_dogs"]}
    if status == "unknown" or "" in toxic_flags:
        return "unknown"
    if status == "aspca_genus_inferred":
        return "likely_toxic"
    return "toxic" if "1" in toxic_flags else "non_toxic"


def pet_summary(row: dict) -> str:
    return {
        "toxic": "toxic to cats and dogs (ASPCA)",
        "non_toxic": "non-toxic to cats and dogs (ASPCA)",
        "likely_toxic": "treat as toxic to pets (related plants are toxic)",
        "unknown": "no verified toxicity record, so keep away from pets",
    }[toxicity(row)]


def watering_hint(days: int) -> str:
    if days >= 14:
        return "let the soil dry out almost completely before watering again."
    if days <= 5:
        return "it likes evenly moist (never soggy) soil, so water when the top centimetre feels dry."
    return "water when the top 2-5 cm of soil feels dry."


def names(row: dict) -> tuple[str, str, list[str]]:
    common = row["common_name"]
    return common, row["scientific_name"], PLANT_ALIASES.get(common, [])


def base_keywords(row: dict) -> list[str]:
    common, scientific, aliases = names(row)
    return [common.lower(), scientific.lower(), *aliases]


def entry_id(row: dict, kind: str) -> str:
    return f"kg_{slugify(row['common_name'])}_{kind}"


def make_overview(row: dict) -> dict:
    common, scientific, aliases = names(row)
    days = int(row["watering_frequency_days"])
    note = ""
    if row["care_confidence"] == "medium":
        note = "\n(These care values are estimated from the plant's family, so treat them as a starting point.)"
    answer = (
        f"{common} ({scientific}) care summary:\n"
        f"- Light: {row['light_requirement_level'].lower()} light "
        f"(about {int(row['min_lux']):,}-{int(row['max_lux']):,} lux).\n"
        f"- Water: about every {days} days as a rough guide; {watering_hint(days)}\n"
        f"- Temperature: {row['min_temp_celsius']}-{row['max_temp_celsius']} °C.\n"
        f"- Humidity: around {row['ideal_humidity_percent']}%.\n"
        f"- Pets: {pet_summary(row)}."
        f"{note}"
    )
    alternatives = [f"How to look after {an(common)}?", f"{common} care guide", f"Care tips for {scientific}"]
    alternatives += [f"How do I take care of my {alias}?" for alias in aliases[:2]]
    return {
        "id": entry_id(row, "care"),
        "topic": "plant_profiles",
        "question": f"How do I care for {an(common)}?",
        "alternatives": alternatives,
        "keywords": base_keywords(row) + ["care guide", "care tips"],
        "answer": answer,
    }


def make_light(row: dict) -> dict:
    common, scientific, aliases = names(row)
    answer = (
        f"The {common} prefers {LIGHT_PHRASES[row['light_requirement_level']]}. "
        f"FloraDB lists roughly {int(row['min_lux']):,}-{int(row['max_lux']):,} lux as its range. "
        "Leaning, stretched growth means it wants more light; bleached or scorched patches mean the sun is too strong."
    )
    alternatives = [
        f"How much sunlight does {an(common)} need?",
        f"Where should I put or keep my {common}?",
        f"Can {an(common)} grow in low light?",
        f"Does {scientific} need direct sun?",
    ]
    alternatives += [f"How much light does {an(alias)} need?" for alias in aliases[:2]]
    return {
        "id": entry_id(row, "light"),
        "topic": "plant_profiles",
        "question": f"How much light does {an(common)} need?",
        "alternatives": alternatives,
        "keywords": base_keywords(row) + ["light", "sunlight", "placement"],
        "answer": answer,
    }


def make_watering(row: dict) -> dict:
    common, scientific, aliases = names(row)
    days = int(row["watering_frequency_days"])
    answer = (
        f"FloraDB gives a typical interval of about every {days} days for the {common}. "
        f"Use that only as a rough guide: {watering_hint(days)} "
        "Water less in winter and in low light, and always use a pot with drainage holes."
    )
    alternatives = [
        f"How much water does {an(common)} need?",
        f"When should I water my {common}?",
        f"Watering schedule for {scientific}",
    ]
    alternatives += [f"How often should I water {an(alias)}?" for alias in aliases[:2]]
    return {
        "id": entry_id(row, "water"),
        "topic": "plant_profiles",
        "question": f"How often should I water {an(common)}?",
        "alternatives": alternatives,
        "keywords": base_keywords(row) + ["watering", "water frequency"],
        "answer": answer,
    }


def make_pet_safety(row: dict) -> dict:
    common, scientific, aliases = names(row)
    status = toxicity(row)
    if status == "toxic":
        answer = (
            f"No. The ASPCA lists the {common} ({scientific}) as toxic to cats and dogs. "
            f"Possible symptoms: {row['toxicity_symptoms'].rstrip('.')}. "
            "Keep it out of reach, and contact your vet if a pet chews it."
        )
    elif status == "non_toxic":
        answer = (
            f"Yes. The ASPCA lists the {common} ({scientific}) as non-toxic to cats and dogs. "
            "Eating any plant can still upset a pet's stomach, so it is best kept out of reach."
        )
    elif status == "likely_toxic":
        answer = (
            f"Treat the {common} ({scientific}) as toxic. There is no ASPCA record for this exact species, "
            "but other plants in the same genus are toxic to cats and dogs. "
            "Keep it away from pets, and contact your vet if one chews it."
        )
    else:
        answer = (
            f"There is no verified toxicity record for the {common} ({scientific}) in the dataset, "
            "so it is not safe to assume it is harmless. Keep it out of reach of pets and ask your vet if one chews it."
        )
    alternatives = [
        f"Is {an(common)} toxic to cats?",
        f"Is {common} poisonous to dogs?",
        f"Is {an(common)} dangerous for pets?",
        f"Is {scientific} pet friendly?",
    ]
    alternatives += [f"Is {alias} safe for pets?" for alias in aliases[:2]]
    return {
        "id": entry_id(row, "pets"),
        "topic": "pet_safety",
        "question": f"Is {an(common)} safe for cats and dogs?",
        "alternatives": alternatives,
        "keywords": base_keywords(row) + ["toxic", "pet safe", "cats", "dogs"],
        "answer": answer,
    }


def finish(entry: dict, row: dict) -> dict:
    """Add the fields shared by every generated entry, in a fixed key order."""
    return {
        "id": entry["id"],
        "topic": entry["topic"],
        "question": entry["question"],
        "alternatives": entry["alternatives"],
        "keywords": entry["keywords"],
        "answer": entry["answer"],
        "source": SOURCE,
        "plant": row["common_name"],
        "scientific_name": row["scientific_name"],
        "data_quality": {
            "care_confidence": row["care_confidence"],
            "toxicity_status": row["toxicity_status"],
        },
        "related": [],
    }


def plant_entries(row: dict) -> list[dict]:
    makers = [make_pet_safety]
    if row["care_confidence"] in CARE_CONFIDENCE_USED:
        makers = [make_overview, make_light, make_watering, make_pet_safety]
    entries = [finish(make(row), row) for make in makers]
    ids = [entry["id"] for entry in entries]
    for entry in entries:
        entry["related"] = [other for other in ids if other != entry["id"]]
    return entries


def list_entry(entry_id_: str, topic: str, question: str, alternatives: list[str],
               keywords: list[str], intro: str, plants: list[str], outro: str) -> dict:
    return {
        "id": entry_id_,
        "topic": topic,
        "question": question,
        "alternatives": alternatives,
        "keywords": keywords,
        "answer": f"{intro} {', '.join(plants)}. {outro}",
        "source": SOURCE,
        "plant": None,
        "scientific_name": None,
        "data_quality": {"care_confidence": "high", "toxicity_status": "aspca_verified"},
        "related": [],
    }


def aggregate_entries(rows: list[dict]) -> list[dict]:
    """Recommendation lists computed from the table (only high-confidence rows)."""
    trusted = [row for row in rows if row["care_confidence"] == "high"]
    pet_safe = [row["common_name"] for row in trusted if toxicity(row) == "non_toxic"]
    low_light = [row["common_name"] for row in trusted if row["light_requirement_level"] in LOW_LIGHT_LEVELS]
    low_water = [row["common_name"] for row in trusted if int(row["watering_frequency_days"]) >= 14]
    humid = [row["common_name"] for row in trusted if int(row["ideal_humidity_percent"]) >= 65]
    sunny = [row["common_name"] for row in trusted if row["light_requirement_level"] == "Direct Sun"]
    return [
        list_entry(
            "kg_list_pet_safe", "pet_safety",
            "Which houseplants are safe for cats and dogs?",
            ["What plants are pet friendly?", "Which plants are safe for pets?",
             "Non-toxic plants for a home with cats", "Which indoor plants can I keep with my dog?",
             "Plants that are not poisonous to pets"],
            ["pet safe", "non toxic", "pet friendly", "safe for cats", "safe for dogs"],
            "According to ASPCA data in the FloraDB dataset, these houseplants are non-toxic to cats and dogs:",
            pet_safe,
            "Any plant can still upset a pet's stomach if eaten in quantity, so keep an eye on chewers.",
        ),
        list_entry(
            "kg_list_low_light", "plant_recommendations",
            "Which plants grow well in low light?",
            ["What plants can live in a dark room?", "Which houseplants tolerate shade?",
             "Plants for a dim corner", "Best plants for a room with little sunlight"],
            ["low light", "dark room", "shade tolerant", "dim corner", "little sunlight"],
            "From the FloraDB dataset, these plants tolerate low or medium light:",
            low_light,
            "Tolerating low light is not the same as loving it: they grow slowly there and need less water. "
            "No plant survives in complete darkness, so a windowless room needs a grow light.",
        ),
        list_entry(
            "kg_list_low_water", "plant_recommendations",
            "Which plants need very little watering?",
            ["What plants are drought tolerant?", "Plants I can forget to water",
             "Low maintenance plants that rarely need water", "Best plants for someone who travels a lot"],
            ["drought tolerant", "little water", "rarely water", "forget to water", "low watering"],
            "These plants in the FloraDB dataset typically go two weeks or more between waterings:",
            low_water,
            "Even drought-tolerant plants rot if they sit in wet soil, so use free-draining mix.",
        ),
        list_entry(
            "kg_list_high_humidity", "plant_recommendations",
            "Which plants need high humidity?",
            ["What plants like humid air?", "Good plants for a bathroom",
             "Which houseplants need moist air?"],
            ["high humidity", "humid", "bathroom plants", "moist air", "tropical"],
            "These plants in the FloraDB dataset prefer humidity of 65% or more:",
            humid,
            "Group them together, use a pebble tray or a humidifier, or keep them in a bright bathroom.",
        ),
        list_entry(
            "kg_list_direct_sun", "plant_recommendations",
            "Which houseplants like direct sun?",
            ["What plants can sit in a sunny window?", "Plants for a south facing window",
             "Which plants love full sun?"],
            ["direct sun", "full sun", "sunny window", "south facing", "lots of sun"],
            "These plants in the FloraDB dataset want direct sun:",
            sunny,
            "Introduce them to strong sun gradually to avoid scorching.",
        ),
    ]


def build(rows: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for row in rows:  # dataset order: common, well-documented plants come first
        entries.extend(plant_entries(row))
    entries.extend(aggregate_entries(rows))
    return entries


def to_json(entries: list[dict]) -> str:
    return json.dumps(entries, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    rows = load_rows(config.RAW_DATASET_PATH)
    entries = build(rows)
    # newline="\n" keeps LF line endings on every OS, so the file is identical on macOS and Windows.
    config.GENERATED_KB_PATH.write_text(to_json(entries), encoding="utf-8", newline="\n")
    care_rows = sum(1 for row in rows if row["care_confidence"] in CARE_CONFIDENCE_USED)
    print(f"Read {len(rows)} plants ({care_rows} with trusted care data).")
    print(f"Wrote {len(entries)} entries to {config.GENERATED_KB_PATH.relative_to(BACKEND_DIR)}")


if __name__ == "__main__":
    main()
