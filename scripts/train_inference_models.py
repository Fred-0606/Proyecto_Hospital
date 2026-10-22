"""Entrena y serializa los pipelines de inferencia desde la raíz del proyecto."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import train_and_serialize_models  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(train_and_serialize_models(), ensure_ascii=False, indent=2))
