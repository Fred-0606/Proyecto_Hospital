"""Pruebas de los modelos baseline de la etapa 06."""

import numpy as np
import pandas as pd

from src.models import (
    CLASSIFICATION_THRESHOLD,
    L2_C,
    L2_RATIO,
    L2_TOLERANCE,
    build_l2_logistic_baseline,
    calculate_binary_metrics,
    summarize_baseline_metrics,
)
from src.preprocessing import build_model_a_preprocessor
from src.model_selection import summarize_comparison_metrics


def test_l2_baseline_has_predefined_configuration() -> None:
    """El baseline debe usar L2 sin ajuste implícito de parámetros."""
    pipeline = build_l2_logistic_baseline(build_model_a_preprocessor())
    classifier = pipeline.named_steps["clasificador"]

    assert classifier.solver == "saga"
    assert classifier.C == L2_C == 1.0
    assert classifier.l1_ratio == L2_RATIO == 0.0
    assert classifier.tol == L2_TOLERANCE == 0.001
    assert classifier.class_weight is None
    assert classifier.random_state == 42


def test_binary_metrics_use_probability_for_auc_and_threshold_for_classes() -> None:
    """AUC usa probabilidades y las métricas de clase usan el umbral declarado."""
    labels = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = calculate_binary_metrics(labels, probabilities)

    assert CLASSIFICATION_THRESHOLD == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["verdaderos_negativos"] == 1
    assert metrics["falsos_positivos"] == 1
    assert metrics["falsos_negativos"] == 1
    assert metrics["verdaderos_positivos"] == 1
    assert metrics["roc_auc"] == 0.75


def test_summary_keeps_models_baselines_and_partitions_separate() -> None:
    """Entrenamiento y validación no deben mezclarse al resumir pliegues."""
    rows = []
    for partition in ("entrenamiento", "validacion"):
        for fold, value in ((1, 0.4), (2, 0.6)):
            rows.append(
                {
                    "modelo": "A",
                    "baseline": "regresion_logistica_l2",
                    "particion": partition,
                    "pliegue": fold,
                    "accuracy": value,
                    "recall": value,
                    "precision": value,
                    "f1": value,
                    "roc_auc": value,
                    "pr_auc": value,
                }
            )

    summary = summarize_baseline_metrics(pd.DataFrame(rows))

    assert len(summary) == 2
    assert set(summary["particion"]) == {"entrenamiento", "validacion"}
    assert summary["recall_mean"].eq(0.5).all()
    assert "recall_std" not in summary.columns


def test_comparison_summary_keeps_candidates_and_partitions_separate() -> None:
    """El resumen comparativo debe conservar candidato y partición."""
    rows = []
    for candidate in ("regresion_logistica", "random_forest"):
        for partition in ("entrenamiento", "validacion"):
            for value in (0.4, 0.6):
                rows.append(
                    {
                        "modelo": "A",
                        "candidato": candidate,
                        "particion": partition,
                        **{
                            metric: value
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

    summary = summarize_comparison_metrics(pd.DataFrame(rows))

    assert len(summary) == 4
    assert summary["pr_auc_mean"].eq(0.5).all()
