---
license: cc-by-nc-4.0
pretty_name: FloraDB — Houseplant Care & Pet-Toxicity Dataset (Free Sample)
tags:
- biology
- botany
- houseplants
- pet-safety
- aspca
- gbif
- iot
size_categories:
- n<1K
---

# 🌿 FloraDB — Houseplant Care & Pet-Toxicity Dataset (Free Sample)

> **Full dataset:** [floradb.dataengineered.io](https://floradb.dataengineered.io/) · **$49 one-time** → [Buy on Stripe](https://buy.stripe.com/fZuaEY3kYgDA9vR2I838400) · the same sample on [Kaggle](https://www.kaggle.com/datasets/dataengineered/floradb-houseplants-care-sample)

A **free sample** of FloraDB: a structured dataset that turns subjective houseplant care advice — *"bright indirect light"*, *"water when dry"* — into **quantitative engineering metrics** (Lux thresholds, watering-day intervals, temperature and humidity ranges), joined to **ASPCA dog/cat toxicity** and grounded on the **GBIF Taxonomic Backbone**, with a source URL on every record.

**This sample contains 100 plants across 28 botanical families** — a deliberate mix of verified, genus-inferred, and unknown toxicity so you see the honest quality flags.

The **full dataset covers 270 curated care plants, 891 ASPCA toxicity records, 702 enriched species, and a 20,000+ species GBIF taxonomic index** (SQLite · CSV · JSON).

## Links

* 🔗 **Official Portal:** [floradb](https://floradb.dataengineered.io) — full snapshot $49 one-time
* 🏆 **Kaggle:** [FloraDB Houseplants Care Sample](https://www.kaggle.com/datasets/dataengineered/floradb-houseplants-care-sample)
* 🔄 **Live lookups (self-serve):** [Houseplant Care & Pet-Toxicity Lookup on Apify](https://apify.com/dataengineered/houseplant-care-toxicity-lookup) — GBIF taxonomy + ASPCA toxicity by scientific name, with care metrics for the 100 sample plants only; pay-per-result
* 🌿 **Interactive Explorer:** [Dataset Sample Explorers Space — FloraDB tab](https://huggingface.co/spaces/Ichlibitiche/dataset-sample-explorers)

## Key Columns

| Column | Description |
| :--- | :--- |
| `scientific_name`, `common_name`, `family` | GBIF-accepted canonical identity |
| `light_requirement_level`, `min_lux`, `max_lux` | Light category normalized to a Lux range |
| `watering_frequency_days`, `min/max_temp_celsius`, `ideal_humidity_percent` | Machine-usable care metrics |
| `is_toxic_to_dogs`, `is_toxic_to_cats`, `toxicity_symptoms` | ASPCA-sourced pet toxicity |
| `toxicity_status` | `aspca_verified` · `aspca_genus_inferred` (conservative) · `unknown` — never guessed safe |
| `care_confidence` | `high` (hand-authored) · `medium` (family-normalized) · `low` (generic default) |
| `gbif_usage_key`, `gbif_source_url` | Full taxonomic provenance — every record re-verifiable |

## Quick Start

```python
import pandas as pd

df = pd.read_csv("hf://datasets/Ichlibitiche/floradb-houseplants-care-sample/floradb_sample.csv")
cat_safe = df[(df["is_toxic_to_cats"] == 0) & (df["toxicity_status"] != "unknown")]
print(len(df), "plants,", len(cat_safe), "verified cat-safe")
```

## Use Cases

- Plant-ID and plant-care apps (structured care + pet-safety data)
- IoT smart planters (Lux targets and watering intervals as machine values)
- "Is this plant safe for my cat?" features backed by ASPCA determinations
- ML / RAG corpora over houseplant care and botany

## License & Safety

Sample data: **CC BY-NC 4.0** — attribution, non-commercial. Full dataset commercially licensed at [floradb](https://floradb.dataengineered.io). **Toxicity data is informational, not veterinary advice** — if a pet ingests a plant, contact a vet or the ASPCA Animal Poison Control Center. Contact: `floradb@dataengineered.io`.
