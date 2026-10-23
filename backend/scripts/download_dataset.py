"""Re-download the FloraDB houseplant sample from Hugging Face (optional).

The CSV is already included in data/raw/, so this is only needed if you want
to fetch a fresh copy. No account or API key is required.

    python3 scripts/download_dataset.py
    python3 scripts/build_knowledge_base.py
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

URL = "https://huggingface.co/datasets/Ichlibitiche/floradb-houseplants-care-sample/resolve/main/floradb_sample.csv"


def main() -> None:
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    config.RAW_DATASET_PATH.write_bytes(data)
    print(f"Saved {len(data):,} bytes to {config.RAW_DATASET_PATH.name}")


if __name__ == "__main__":
    main()
