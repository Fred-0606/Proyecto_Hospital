"""Resuelve rutas portables y prepara los directorios de salida del proyecto."""

from pathlib import Path


def get_project_root() -> Path:
    """Devuelve la raíz del repositorio a partir de la ubicación de este módulo."""
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

_OUTPUT_DIRECTORIES = (
    OUTPUTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)

for _directory in _OUTPUT_DIRECTORIES:
    _directory.mkdir(parents=True, exist_ok=True)
