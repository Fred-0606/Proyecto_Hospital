"""Pruebas de las visualizaciones reproducibles."""

from pathlib import Path

import pandas as pd

from src.plots import (
    plot_baseline_confusion_matrices,
    plot_baseline_metric_comparison,
    plot_candidate_metric_comparison,
    plot_null_counts,
    plot_target_balance,
    plot_selected_confusion_matrices,
)


def test_plot_null_counts_creates_nonempty_file(tmp_path: Path) -> None:
    """La gráfica de nulos debe crear una imagen no vacía."""
    dataframe = pd.DataFrame({"completa": [1, 2], "incompleta": [1, None]})
    output_path = tmp_path / "nulos.png"

    result = plot_null_counts(dataframe, output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_baseline_plots_create_nonempty_files(tmp_path: Path) -> None:
    """Las figuras baseline deben generarse a partir de métricas agregadas."""
    fold_metrics = pd.DataFrame(
        {
            "modelo": ["A", "A"],
            "baseline": ["clase_mayoritaria", "regresion_logistica_l2"],
            "verdaderos_negativos": [30, 25],
            "falsos_positivos": [0, 5],
            "falsos_negativos": [20, 8],
            "verdaderos_positivos": [0, 12],
        }
    )
    summary = pd.DataFrame(
        {
            "modelo": ["A", "A"],
            "baseline": ["clase_mayoritaria", "regresion_logistica_l2"],
            **{
                f"{metric}_mean": [0.5, 0.7]
                for metric in (
                    "accuracy",
                    "recall",
                    "precision",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                )
            },
        }
    )
    confusion_path = tmp_path / "confusion.png"
    metrics_path = tmp_path / "metrics.png"

    plot_baseline_confusion_matrices(fold_metrics, confusion_path)
    plot_baseline_metric_comparison(summary, metrics_path)

    assert confusion_path.stat().st_size > 0
    assert metrics_path.stat().st_size > 0


def test_plot_target_balance_creates_nonempty_file(tmp_path: Path) -> None:
    """La gráfica comparativa de objetivos debe crear una imagen no vacía."""
    balance = pd.DataFrame(
        {
            "modelo": ["Modelo A — ingreso"] * 2 + ["Modelo B — salida"] * 2,
            "clase": [0, 1, 0, 1],
            "etiqueta": ["Hasta 6 horas", "Más de 6 horas"] * 2,
            "registros": [40, 60, 70, 30],
            "porcentaje": [40.0, 60.0, 70.0, 30.0],
        }
    )
    output_path = tmp_path / "balance.png"

    result = plot_target_balance(balance, output_path)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_stage_07_plots_create_nonempty_files(tmp_path: Path) -> None:
    """Las figuras comparativas deben aceptar resúmenes y selecciones compactas."""
    summary = pd.DataFrame(
        {
            "modelo": ["A"] * 4,
            "candidato": [
                "regresion_logistica",
                "arbol_decision",
                "random_forest",
                "xgboost",
            ],
            **{
                f"{metric}_mean": [0.5, 0.6, 0.7, 0.8]
                for metric in ("recall", "precision", "f1", "roc_auc", "pr_auc")
            },
        }
    )
    selected = pd.DataFrame(
        {
            "modelo": ["A"],
            "candidato": ["xgboost"],
            "umbral": [0.4],
            "verdaderos_negativos": [25],
            "falsos_positivos": [5],
            "falsos_negativos": [8],
            "verdaderos_positivos": [12],
        }
    )
    comparison_path = tmp_path / "comparison.png"
    confusion_path = tmp_path / "selected_confusion.png"

    plot_candidate_metric_comparison(summary, comparison_path)
    plot_selected_confusion_matrices(selected, confusion_path)

    assert comparison_path.stat().st_size > 0
    assert confusion_path.stat().st_size > 0
