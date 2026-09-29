"""Prepara particiones temporales y transformadores sin fuga de información."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_io import (
    project_relative_path,
    require_file,
    save_dataframe_as_csv,
    save_dataframe_as_parquet,
    save_json,
)
from src.features import MODEL_A_OUTPUT_PATH, MODEL_B_OUTPUT_PATH
from src.paths import METRICS_DIR, PROCESSED_DATA_DIR, TABLES_DIR

TEST_START_DATE = pd.Timestamp("2026-03-01")

# Cada periodo se valida después de entrenar con todos los datos anteriores.
VALIDATION_PERIODS = (
    (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-07-01")),
    (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-01")),
    (pd.Timestamp("2025-09-01"), pd.Timestamp("2025-11-01")),
    (pd.Timestamp("2025-11-01"), pd.Timestamp("2026-01-01")),
    (pd.Timestamp("2026-01-01"), TEST_START_DATE),
)

MODEL_A_TARGET = "estancia_prolongada_a"
MODEL_B_TARGET = "estancia_prolongada_b"

MODEL_A_NUMERIC = [
    "edad",
    "volumen_ingresos_3h",
    "volumen_ingresos_6h",
    "volumen_ingresos_24h",
]
MODEL_A_CATEGORICAL = [
    "triage",
    "afiliacion",
    "hora_ingreso",
    "dia_semana_ingreso",
    "mes_ingreso",
    "fin_semana_ingreso",
]
MODEL_A_PREDICTORS = MODEL_A_NUMERIC + MODEL_A_CATEGORICAL

MODEL_B_NUMERIC = [
    "edad",
    "pacientes_activos_en_ingreso",
    "pacientes_pendientes_de_salida",
]
MODEL_B_DURATION = ["duracion_ingreso_calculada"]
MODEL_B_CATEGORICAL = [
    "triage",
    "afiliacion",
    "sala_observacion",
    "grupo_cama",
    "conducta",
    "hora_ingreso",
    "dia_semana_ingreso",
    "mes_ingreso",
    "fin_semana_ingreso",
    "hora_conducta",
    "dia_semana_conducta",
    "mes_conducta",
    "fin_semana_conducta",
]
MODEL_B_SPECIALTIES = ["especialidades_tratantes"]
MODEL_B_PREDICTORS = (
    MODEL_B_NUMERIC
    + MODEL_B_DURATION
    + MODEL_B_CATEGORICAL
    + MODEL_B_SPECIALTIES
)

MODEL_A_DEVELOPMENT_PATH = PROCESSED_DATA_DIR / "04_ingreso_desarrollo.parquet"
MODEL_A_TEST_PATH = PROCESSED_DATA_DIR / "04_ingreso_prueba.parquet"
MODEL_B_DEVELOPMENT_PATH = PROCESSED_DATA_DIR / "04_salida_desarrollo.parquet"
MODEL_B_TEST_PATH = PROCESSED_DATA_DIR / "04_salida_prueba.parquet"
STAGE_04_SUMMARY_PATH = TABLES_DIR / "04_resumen_particiones.csv"
STAGE_04_METRICS_PATH = METRICS_DIR / "04_particion_temporal.json"


def validate_model_dataset(
    dataframe: pd.DataFrame,
    predictors: list[str],
    target: str,
) -> None:
    """Comprueba el contrato mínimo de una tabla analítica de la etapa 04."""
    required = {"fecha_referencia", target, *predictors}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en la etapa 04: {missing}.")
    if dataframe.empty:
        raise ValueError("La tabla analítica de la etapa 04 está vacía.")
    if dataframe["fecha_referencia"].isna().any():
        raise ValueError("fecha_referencia contiene valores nulos.")
    if not set(dataframe[target].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{target} debe contener únicamente las clases 0 y 1.")
    if dataframe[target].nunique(dropna=True) != 2:
        raise ValueError(f"{target} debe contener ambas clases.")


def temporal_development_test_split(
    dataframe: pd.DataFrame,
    cutoff: pd.Timestamp = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa desarrollo y prueba futura mediante una frontera calendario."""
    ordered = dataframe.sort_values("fecha_referencia", kind="stable").reset_index(
        drop=True
    )
    development = ordered.loc[ordered["fecha_referencia"] < cutoff].copy()
    test = ordered.loc[ordered["fecha_referencia"] >= cutoff].copy()
    if development.empty or test.empty:
        raise ValueError(
            f"El corte {cutoff.date()} debe dejar registros en desarrollo y prueba."
        )
    return development, test


def build_expanding_time_splits(
    development: pd.DataFrame,
    target: str,
    validation_periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (
        VALIDATION_PERIODS
    ),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve pliegues con entrenamiento acumulativo y validación posterior."""
    dates = pd.to_datetime(development["fecha_referencia"])
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for start, end in validation_periods:
        train_indices = np.flatnonzero((dates < start).to_numpy())
        validation_indices = np.flatnonzero(
            ((dates >= start) & (dates < end)).to_numpy()
        )
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(
                f"El pliegue {start.date()}–{end.date()} quedó sin observaciones."
            )
        for name, indices in (
            ("entrenamiento", train_indices),
            ("validación", validation_indices),
        ):
            if development.iloc[indices][target].nunique() != 2:
                raise ValueError(
                    f"El bloque de {name} iniciado en {start.date()} "
                    "no contiene ambas clases."
                )
        splits.append((train_indices, validation_indices))
    return splits


def _flatten_text_column(values: Any) -> np.ndarray:
    """Convierte la columna 2D recibida por sklearn en una secuencia de textos."""
    return np.asarray(values, dtype=object).reshape(-1).astype(str)


def _to_string_matrix(values: Any) -> np.ndarray:
    """Homogeneiza categorías numéricas y textuales antes de codificarlas."""
    return np.asarray(values, dtype=object).astype(str)


def _numeric_pipeline(*, add_indicator: bool = False) -> Pipeline:
    """Crea imputación por mediana y escalamiento ajustados durante fit()."""
    return Pipeline(
        steps=[
            (
                "imputacion",
                SimpleImputer(strategy="median", add_indicator=add_indicator),
            ),
            ("escalamiento", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    """Crea imputación explícita y codificación tolerante a categorías nuevas."""
    return Pipeline(
        steps=[
            (
                "imputacion",
                SimpleImputer(
                    strategy="constant",
                    fill_value="NO_REGISTRADO",
                    missing_values=pd.NA,
                ),
            ),
            (
                "texto",
                FunctionTransformer(_to_string_matrix, validate=False),
            ),
            (
                "codificacion",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )


def _specialty_pipeline() -> Pipeline:
    """Codifica cada especialidad individual del campo multietiqueta."""
    return Pipeline(
        steps=[
            (
                "imputacion",
                SimpleImputer(
                    strategy="constant",
                    fill_value="sin_especialidad",
                    missing_values=pd.NA,
                ),
            ),
            (
                "texto_1d",
                FunctionTransformer(_flatten_text_column, validate=False),
            ),
            (
                "codificacion",
                CountVectorizer(
                    binary=True,
                    lowercase=False,
                    token_pattern=r"[^ |]+",
                ),
            ),
        ]
    )


def build_model_a_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador interpretable del modelo de ingreso."""
    return ColumnTransformer(
        transformers=[
            ("numericas", _numeric_pipeline(), MODEL_A_NUMERIC),
            ("categoricas", _categorical_pipeline(), MODEL_A_CATEGORICAL),
        ],
        remainder="drop",
    )


def build_model_b_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador del modelo de salida y sus multietiquetas."""
    return ColumnTransformer(
        transformers=[
            ("numericas", _numeric_pipeline(), MODEL_B_NUMERIC),
            (
                "duracion_ingreso",
                _numeric_pipeline(add_indicator=True),
                MODEL_B_DURATION,
            ),
            ("categoricas", _categorical_pipeline(), MODEL_B_CATEGORICAL),
            ("especialidades", _specialty_pipeline(), MODEL_B_SPECIALTIES),
        ],
        remainder="drop",
    )


def _partition_summary(
    model: str,
    partition: str,
    dataframe: pd.DataFrame,
    target: str,
    *,
    show_target_prevalence: bool = True,
) -> dict[str, Any]:
    """Resume tamaño y periodo sin revelar el objetivo de la prueba reservada."""
    return {
        "modelo": model,
        "particion": partition,
        "registros": len(dataframe),
        "porcentaje_positivo": (
            round(float(dataframe[target].mean() * 100), 2)
            if show_target_prevalence
            else pd.NA
        ),
        "fecha_inicial": dataframe["fecha_referencia"].min(),
        "fecha_final": dataframe["fecha_referencia"].max(),
    }


def _fold_metadata(
    development: pd.DataFrame,
    target: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Describe los pliegues sin persistir índices fila por fila."""
    metadata: list[dict[str, Any]] = []
    for number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        metadata.append(
            {
                "pliegue": number,
                "entrenamiento_registros": len(train),
                "entrenamiento_inicio": train["fecha_referencia"].min(),
                "entrenamiento_fin": train["fecha_referencia"].max(),
                "entrenamiento_positivos_pct": round(
                    float(train[target].mean() * 100), 2
                ),
                "validacion_registros": len(validation),
                "validacion_inicio": validation["fecha_referencia"].min(),
                "validacion_fin": validation["fecha_referencia"].max(),
                "validacion_positivos_pct": round(
                    float(validation[target].mean() * 100), 2
                ),
            }
        )
    return metadata


def run_stage_04_partitioning(
    model_a_path: Path = MODEL_A_OUTPUT_PATH,
    model_b_path: Path = MODEL_B_OUTPUT_PATH,
) -> dict[str, Any]:
    """Ejecuta la etapa 05 y guarda solo particiones y evidencia necesaria."""
    require_file(
        model_a_path,
        "Ejecute primero la etapa 04 para generar 04_modelo_a.parquet.",
    )
    require_file(
        model_b_path,
        "Ejecute primero la etapa 04 para generar 04_modelo_b.parquet.",
    )
    model_a = pd.read_parquet(model_a_path)
    model_b = pd.read_parquet(model_b_path)
    validate_model_dataset(model_a, MODEL_A_PREDICTORS, MODEL_A_TARGET)
    validate_model_dataset(model_b, MODEL_B_PREDICTORS, MODEL_B_TARGET)

    a_development, a_test = temporal_development_test_split(model_a)
    b_development, b_test = temporal_development_test_split(model_b)
    a_splits = build_expanding_time_splits(a_development, MODEL_A_TARGET)
    b_splits = build_expanding_time_splits(b_development, MODEL_B_TARGET)

    output_frames = (
        (a_development, MODEL_A_DEVELOPMENT_PATH),
        (a_test, MODEL_A_TEST_PATH),
        (b_development, MODEL_B_DEVELOPMENT_PATH),
        (b_test, MODEL_B_TEST_PATH),
    )
    for dataframe, path in output_frames:
        save_dataframe_as_parquet(dataframe, path)

    summary = pd.DataFrame(
        [
            _partition_summary("A", "desarrollo", a_development, MODEL_A_TARGET),
            _partition_summary(
                "A",
                "prueba",
                a_test,
                MODEL_A_TARGET,
                show_target_prevalence=False,
            ),
            _partition_summary("B", "desarrollo", b_development, MODEL_B_TARGET),
            _partition_summary(
                "B",
                "prueba",
                b_test,
                MODEL_B_TARGET,
                show_target_prevalence=False,
            ),
        ]
    )
    save_dataframe_as_csv(summary, STAGE_04_SUMMARY_PATH)

    metadata = {
        "fecha_inicio_prueba": TEST_START_DATE,
        "regla_particion": "desarrollo < fecha_inicio_prueba; prueba >= fecha_inicio_prueba",
        "estadisticas_objetivo_prueba_ocultas": True,
        "advertencia_auditoria": (
            "La prevalencia de prueba fue observada en ejecuciones anteriores, "
            "pero no se utilizó para ajustar transformadores, modelos o umbrales."
        ),
        "predictores_modelo_a": MODEL_A_PREDICTORS,
        "predictores_modelo_b": MODEL_B_PREDICTORS,
        "pliegues_modelo_a": _fold_metadata(
            a_development, MODEL_A_TARGET, a_splits
        ),
        "pliegues_modelo_b": _fold_metadata(
            b_development, MODEL_B_TARGET, b_splits
        ),
        "artefactos": [
            project_relative_path(path)
            for _, path in output_frames
        ]
        + [project_relative_path(STAGE_04_SUMMARY_PATH)],
    }
    save_json(metadata, STAGE_04_METRICS_PATH)
    return {
        "modelo_a_desarrollo": a_development,
        "modelo_a_prueba": a_test,
        "modelo_b_desarrollo": b_development,
        "modelo_b_prueba": b_test,
        "pliegues_modelo_a": a_splits,
        "pliegues_modelo_b": b_splits,
        "resumen": summary,
        "metadatos": metadata,
    }
