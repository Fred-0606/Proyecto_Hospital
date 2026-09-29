"""Ejecuta una etapa Quarto individual desde la raíz del repositorio."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STAGES = {
    "00": "00_contexto_y_objetivos.qmd",
    "01": "01_etl_extraccion_anonimizacion.qmd",
    "02": "02_diseno_analitico_y_seleccion.qmd",
    "03": "03_etl_transformacion_calidad.qmd",
    "04": "04_conjuntos_datos_y_particion_temporal.qmd",
    "05": "05_eda_desarrollo.qmd",
    "06": "06_feature_engineering_y_preprocesamiento.qmd",
    "07": "07_modelo_baseline.qmd",
    "08": "08_modelos_comparativos.qmd",
    "09": "09_evaluacion_final.qmd",
    "10": "10_interpretabilidad.qmd",
    "11": "11_conclusiones.qmd",
}


def main() -> int:
    """Valida el número solicitado y ejecuta la etapa en una sesión limpia."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=PIPELINE_STAGES)
    args = parser.parse_args()

    python_path = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_path.is_file():
        raise FileNotFoundError(
            "No existe .venv\\Scripts\\python.exe. Cree primero el entorno virtual."
        )

    stage_path = PROJECT_ROOT / "pipeline" / PIPELINE_STAGES[args.stage]
    environment = os.environ.copy()
    environment["QUARTO_PYTHON"] = str(python_path)
    result = subprocess.run(
        ["quarto", "render", str(stage_path)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
