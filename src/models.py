"""Entrena los modelos baseline mediante validación temporal sin abrir prueba."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_io import require_file, save_dataframe_as_csv, save_json
from src.paths import FIGURES_DIR, METRICS_DIR, TABLES_DIR
from src.plots import plot_baseline_confusion_matrices, plot_baseline_metric_comparison
from src.preprocessing import (
    MODEL_A_DEVELOPMENT_PATH,
    MODEL_A_PREDICTORS,
    MODEL_A_TARGET,
    MODEL_B_DEVELOPMENT_PATH,
    MODEL_B_PREDICTORS,
    MODEL_B_TARGET,
    build_expanding_time_splits,
    build_model_a_preprocessor,
    build_model_b_preprocessor,
)

RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5
L2_C = 1.0
L2_RATIO = 0.0
L2_TOLERANCE = 0.001
L2_MAX_ITER = 1000

STAGE_07_FOLDS_PATH = TABLES_DIR / "07_metricas_baseline_por_pliegue.csv"
STAGE_07_SUMMARY_PATH = TABLES_DIR / "07_resumen_baseline.csv"
STAGE_07_METRICS_PATH = METRICS_DIR / "07_modelos_baseline.json"
STAGE_07_CONFUSION_FIGURE_PATH = FIGURES_DIR / "07_matrices_confusion_baseline.png"
STAGE_07_METRICS_FIGURE_PATH = FIGURES_DIR / "07_comparacion_metricas_baseline.png"


def build_l2_logistic_baseline(
    preprocessor: object,
) -> Pipeline:
    """Construye la regresión logística L2 preespecificada."""
    classifier = LogisticRegression(
        solver="saga",
        l1_ratio=L2_RATIO,
        C=L2_C,
        tol=L2_TOLERANCE,
        max_iter=L2_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocessor),
            ("clasificador", classifier),
        ]
    )


def calculate_binary_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, float | int]:
    """Calcula métricas binarias y componentes de la matriz de confusión."""
    predictions = (probabilities >= threshold).astype("int8")
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "verdaderos_negativos": int(true_negative),
        "falsos_positivos": int(false_positive),
        "falsos_negativos": int(false_negative),
        "verdaderos_positivos": int(true_positive),
    }


def evaluate_baselines_on_temporal_folds(
    development: pd.DataFrame,
    predictors: list[str],
    target: str,
    model_name: str,
    preprocessor_builder: Callable[[], object],
) -> pd.DataFrame:
    """Evalúa referencia mayoritaria y regresión L2 en cada pliegue temporal."""
    rows: list[dict[str, object]] = []
    splits = build_expanding_time_splits(development, target)

    for fold_number, (train_indices, validation_indices) in enumerate(
        splits,
        start=1,
    ):
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        x_train = train[predictors]
        y_train = train[target]
        x_validation = validation[predictors]
        y_validation = validation[target]

        candidates = {
            "clase_mayoritaria": DummyClassifier(strategy="most_frequent"),
            "regresion_logistica_l2": build_l2_logistic_baseline(
                preprocessor_builder()
            ),
        }
        for baseline_name, estimator in candidates.items():
            fitted_estimator = clone(estimator).fit(x_train, y_train)
            if baseline_name == "regresion_logistica_l2":
                classifier = fitted_estimator.named_steps["clasificador"]
                iterations = int(classifier.n_iter_[0])
                converged = iterations < classifier.max_iter
            else:
                iterations = pd.NA
                converged = pd.NA
            for partition, features, labels in (
                ("entrenamiento", x_train, y_train),
                ("validacion", x_validation, y_validation),
            ):
                probabilities = fitted_estimator.predict_proba(features)[:, 1]
                rows.append(
                    {
                        "modelo": model_name,
                        "baseline": baseline_name,
                        "pliegue": fold_number,
                        "particion": partition,
                        "registros": len(labels),
                        "prevalencia": float(labels.mean()),
                        "iteraciones": iterations,
                        "convergencia_alcanzada": converged,
                        **calculate_binary_metrics(labels, probabilities),
                    }
                )
    return pd.DataFrame(rows)


def summarize_baseline_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume el promedio de las métricas entre pliegues."""
    metric_columns = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc"]
    return (
        fold_metrics.groupby(["modelo", "baseline", "particion"], as_index=False)[
            metric_columns
        ]
        .mean()
        .rename(columns={metric: f"{metric}_mean" for metric in metric_columns})
    )


def run_stage_07_baselines() -> dict[str, object]:
    """Ejecuta los baselines de A y B usando solo sus datos de desarrollo."""
    require_file(
        MODEL_A_DEVELOPMENT_PATH,
        "Ejecute primero la etapa 04 para generar 04_ingreso_desarrollo.parquet.",
    )
    require_file(
        MODEL_B_DEVELOPMENT_PATH,
        "Ejecute primero la etapa 04 para generar 04_salida_desarrollo.parquet.",
    )
    model_a_development = pd.read_parquet(MODEL_A_DEVELOPMENT_PATH)
    model_b_development = pd.read_parquet(MODEL_B_DEVELOPMENT_PATH)

    fold_metrics = pd.concat(
        [
            evaluate_baselines_on_temporal_folds(
                model_a_development,
                MODEL_A_PREDICTORS,
                MODEL_A_TARGET,
                "A",
                build_model_a_preprocessor,
            ),
            evaluate_baselines_on_temporal_folds(
                model_b_development,
                MODEL_B_PREDICTORS,
                MODEL_B_TARGET,
                "B",
                build_model_b_preprocessor,
            ),
        ],
        ignore_index=True,
    )
    return finalize_stage_07_results(fold_metrics)


def finalize_stage_07_results(fold_metrics: pd.DataFrame) -> dict[str, object]:
    """Resume, visualiza y guarda resultados baseline ya calculados."""
    summary = summarize_baseline_metrics(fold_metrics)
    save_dataframe_as_csv(fold_metrics, STAGE_07_FOLDS_PATH)
    save_dataframe_as_csv(summary, STAGE_07_SUMMARY_PATH)
    validation_metrics = fold_metrics.loc[
        fold_metrics["particion"].eq("validacion")
    ].copy()
    plot_baseline_confusion_matrices(
        validation_metrics, STAGE_07_CONFUSION_FIGURE_PATH
    )
    plot_baseline_metric_comparison(
        summary.loc[summary["particion"].eq("validacion")],
        STAGE_07_METRICS_FIGURE_PATH,
    )
    metadata = {
        "prueba_final_utilizada": False,
        "pliegues_temporales": 5,
        "umbral_clasificacion": CLASSIFICATION_THRESHOLD,
        "referencia_ingenua": {"strategy": "most_frequent"},
        "regresion_logistica": {
            "penalty": "l2",
            "solver": "saga",
            "C": L2_C,
            "l1_ratio": L2_RATIO,
            "tol": L2_TOLERANCE,
            "max_iter": L2_MAX_ITER,
            "class_weight": None,
            "random_state": RANDOM_STATE,
        },
        "artefactos": [
            STAGE_07_FOLDS_PATH.name,
            STAGE_07_SUMMARY_PATH.name,
            STAGE_07_CONFUSION_FIGURE_PATH.name,
            STAGE_07_METRICS_FIGURE_PATH.name,
        ],
    }
    save_json(metadata, STAGE_07_METRICS_PATH)
    return {
        "metricas_por_pliegue": fold_metrics,
        "resumen": summary,
        "metadatos": metadata,
    }
