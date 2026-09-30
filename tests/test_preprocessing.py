"""Pruebas de partición temporal y transformadores de la etapa 05."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    MODEL_A_PREDICTORS,
    MODEL_A_TARGET,
    MODEL_B_PREDICTORS,
    MODEL_B_TARGET,
    build_expanding_time_splits,
    build_model_a_preprocessor,
    build_model_b_preprocessor,
    _partition_summary,
    run_stage_04_partitioning,
    temporal_development_test_split,
    validate_model_dataset,
)


def test_temporal_split_respects_cutoff_and_keeps_equal_dates_together() -> None:
    """Una misma fecha no puede aparecer en lados distintos de la frontera."""
    dataframe = pd.DataFrame(
        {
            "fecha_referencia": pd.to_datetime(
                ["2026-03-01", "2026-02-28", "2026-03-01", "2026-01-01"]
            ),
            "valor": [1, 2, 3, 4],
        }
    )

    development, test = temporal_development_test_split(dataframe)

    assert development["fecha_referencia"].max() < pd.Timestamp("2026-03-01")
    assert test["fecha_referencia"].min() >= pd.Timestamp("2026-03-01")
    assert len(test) == 2


def test_test_partition_summary_hides_target_prevalence() -> None:
    """El resumen previo a la etapa 09 no debe revelar el objetivo de prueba."""
    test = pd.DataFrame(
        {
            "fecha_referencia": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            MODEL_A_TARGET: [0, 1],
        }
    )

    summary = _partition_summary(
        "A",
        "prueba",
        test,
        MODEL_A_TARGET,
        show_target_prevalence=False,
    )

    assert pd.isna(summary["porcentaje_positivo"])


def test_expanding_splits_accumulate_past_periods() -> None:
    """Cada entrenamiento debe crecer y finalizar antes de su validación."""
    development = pd.DataFrame(
        {
            "fecha_referencia": pd.date_range("2024-09-01", "2026-02-01", freq="MS")
            .repeat(2),
            "objetivo": [0, 1] * 18,
        }
    )

    splits = build_expanding_time_splits(development, "objetivo")

    assert len(splits) == 5
    previous_train_size = 0
    for train_indices, validation_indices in splits:
        train = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        assert len(train) > previous_train_size
        assert train["fecha_referencia"].max() < validation["fecha_referencia"].min()
        assert set(train["objetivo"]) == {0, 1}
        assert set(validation["objetivo"]) == {0, 1}
        previous_train_size = len(train)


def test_model_a_preprocessor_learns_median_only_from_fit_data() -> None:
    """La mediana de edad debe ignorar valores extremos usados solo al transformar."""
    train = pd.DataFrame(
        {
            "edad": [20.0, 40.0, np.nan],
            "volumen_ingresos_3h": [2, 3, 4],
            "volumen_ingresos_6h": [4, 6, 8],
            "volumen_ingresos_24h": [20, 25, 30],
            "triage": ["t2", "t3", "t2"],
            "afiliacion": ["a", "a", "b"],
            "hora_ingreso": [8, 9, 10],
            "dia_semana_ingreso": ["lunes", "martes", "miercoles"],
            "mes_ingreso": ["enero", "enero", "enero"],
            "fin_semana_ingreso": [0, 0, 0],
        }
    )
    validation = train.iloc[[2]].copy()
    validation.loc[:, "edad"] = 1000.0
    preprocessor = build_model_a_preprocessor().fit(train)

    median = preprocessor.named_transformers_["numericas"].named_steps[
        "imputacion"
    ].statistics_[0]
    transformed = preprocessor.transform(validation)

    assert median == 30.0
    assert transformed.shape[0] == 1


def test_model_b_preprocessor_handles_new_categories() -> None:
    """Las categorías futuras deben transformarse sin modificar el ajuste."""
    train = pd.DataFrame(
        {
            "edad": [20.0, np.nan],
            "pacientes_activos_en_ingreso": [5, 7],
            "pacientes_pendientes_de_salida": [3, 4],
            "duracion_ingreso_calculada": [3.0, np.nan],
            "triage": ["t2", "t3"],
            "afiliacion": ["a", "b"],
            "sala_observacion": pd.Series(["s1", pd.NA], dtype="string"),
            "grupo_cama": ["a_cama", "b_cama"],
            "conducta": ["alta", "hospitalizacion"],
            "hora_ingreso": [8, 9],
            "dia_semana_ingreso": ["lunes", "martes"],
            "mes_ingreso": ["enero", "enero"],
            "fin_semana_ingreso": [0, 0],
            "hora_conducta": [12, 13],
            "dia_semana_conducta": ["lunes", "martes"],
            "mes_conducta": ["enero", "enero"],
            "fin_semana_conducta": [0, 0],
        }
    )
    future = train.iloc[[0]].copy()
    future.loc[:, "triage"] = "triage_nuevo"
    preprocessor = build_model_b_preprocessor().fit(train)

    transformed = preprocessor.transform(future)
    duration_imputer = preprocessor.named_transformers_[
        "duracion_ingreso"
    ].named_steps["imputacion"]
    assert transformed.shape[0] == 1
    assert duration_imputer.add_indicator is True
    assert np.isfinite(transformed.data).all()


def test_dataset_contract_rejects_missing_predictor() -> None:
    """El contrato debe identificar claramente una columna ausente."""
    dataframe = pd.DataFrame(
        {
            "fecha_referencia": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            MODEL_A_TARGET: [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_model_dataset(dataframe, MODEL_A_PREDICTORS, MODEL_A_TARGET)


def test_stage_04_requires_analytical_files(tmp_path: Path) -> None:
    """La partición debe indicar que primero se construyen los conjuntos A y B."""
    with pytest.raises(FileNotFoundError, match="Ejecute primero la etapa 04"):
        run_stage_04_partitioning(
            model_a_path=tmp_path / "modelo_a.parquet",
            model_b_path=tmp_path / "modelo_b.parquet",
        )


def test_predictor_lists_exclude_dates_targets_and_outcome_durations() -> None:
    """Las variables de control o resultado no pueden entrar al preprocesador."""
    assert "fecha_referencia" not in MODEL_A_PREDICTORS
    assert MODEL_A_TARGET not in MODEL_A_PREDICTORS
    assert "duracion_ingreso_calculada" not in MODEL_A_PREDICTORS
    assert "fecha_referencia" not in MODEL_B_PREDICTORS
    assert MODEL_B_TARGET not in MODEL_B_PREDICTORS
    assert "duracion_salida_calculada" not in MODEL_B_PREDICTORS
    assert "especialidades_tratantes" not in MODEL_B_PREDICTORS
