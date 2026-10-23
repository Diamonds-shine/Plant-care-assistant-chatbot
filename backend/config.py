"""Central configuration for the GreenBuddy backend.

Every tunable value lives here so it can be changed without touching the
NLP or API code. Paths are built with pathlib so they work on macOS,
Linux and Windows alike.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Knowledge base files
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CORE_KB_PATH = KNOWLEDGE_BASE_DIR / "plant_care_core.json"
GENERATED_KB_PATH = KNOWLEDGE_BASE_DIR / "plant_care_generated.json"
RAW_DATASET_PATH = BASE_DIR / "data" / "raw" / "floradb_houseplants_sample.csv"

# ---------------------------------------------------------------------------
# Retrieval thresholds (cosine similarity, range 0..1)
#
# score >= HIGH_CONFIDENCE_THRESHOLD           -> confident answer
# LOW_CONFIDENCE_THRESHOLD <= score < HIGH     -> answer, flagged as "closest match"
# score <  LOW_CONFIDENCE_THRESHOLD            -> fallback message
#
# LOW_CONFIDENCE_THRESHOLD was chosen by running `python3 -m nlp.evaluator`
# on the calibration questions (0.10-0.30 scored equally; the highest, safest
# value was kept). HIGH_CONFIDENCE_THRESHOLD only changes how the answer is
# labelled in the UI. Neither is a universal constant.
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------
MAX_QUESTION_LENGTH = 500

# ---------------------------------------------------------------------------
# Server
# Port 5001 is used instead of Flask's default 5000 because macOS
# (Monterey and later) reserves port 5000 for the AirPlay Receiver.
# ---------------------------------------------------------------------------
HOST = os.environ.get("GREENBUDDY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GREENBUDDY_PORT", "5001"))

# The web page is served by Flask itself (same origin), so CORS is only
# needed if the frontend is opened from another local server. Keep it narrow.
FRONTEND_DIR = BASE_DIR.parent / "frontend"
_default_origins = "http://localhost:5001,http://127.0.0.1:5001"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("GREENBUDDY_CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

FALLBACK_MESSAGE = (
    "I'm best at plant-care questions. Try asking me about watering, sunlight, "
    "soil, leaves, pests, or a specific houseplant like a snake plant or pothos!"
)

NO_TERMS_MESSAGE = (
    "I couldn't find any plant-care words in that message. Could you ask a "
    "question such as \"Why are my leaves turning yellow?\""
)
