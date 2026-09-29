"""Define artefactos y resúmenes transparentes para la comparación de modelos."""

from __future__ import annotations

import pandas as pd

from src.paths import FIGURES_DIR, METRICS_DIR, TABLES_DIR

STAGE_08_SEARCH_PATH = TABLES_DIR / "08_busqueda_hiperparametros.csv"
STAGE_08_FOLDS_PATH = TABLES_DIR / "08_metricas_modelos_por_pliegue.csv"
STAGE_08_SUMMARY_PATH = TABLES_DIR / "08_resumen_modelos.csv"
STAGE_08_THRESHOLDS_PATH = TABLES_DIR / "08_umbrales_seleccionados.csv"
STAGE_08_SELECTION_PATH = METRICS_DIR / "08_seleccion_modelos.json"
STAGE_08_METRICS_FIGURE_PATH = FIGURES_DIR / "08_comparacion_modelos.png"
STAGE_08_CONFUSION_FIGURE_PATH = FIGURES_DIR / "08_matrices_confusion_seleccionados.png"

COMPARISON_METRICS = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc"]


def summarize_comparison_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcula medias por tarea, candidato y partición sin ocultar el entrenamiento."""
    return (
        fold_metrics.groupby(
            ["modelo", "candidato", "particion"],
            as_index=False,
        )[COMPARISON_METRICS]
        .mean()
        .rename(
            columns={metric: f"{metric}_mean" for metric in COMPARISON_METRICS}
        )
    )
