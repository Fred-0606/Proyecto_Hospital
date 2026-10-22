"""Pruebas de la capa de inferencia serializada."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.inference import (
    build_episode_predictors,
    build_new_patient_predictors,
    predict_patient,
)


class ProbabilityStub:
    """Pipeline mínimo determinista para probar la regla del umbral."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, rows: Any) -> np.ndarray:
        """Devuelve una probabilidad binaria para la única fila esperada."""
        assert len(rows) == 1
        return np.array([[1 - self.probability, self.probability]])


def test_prediction_returns_yes_at_or_above_threshold() -> None:
    """Convierte la probabilidad en una decisión clínica explícita."""
    metadata = {
        "models": {
            "A": {
                "predictors": ["edad"],
                "threshold": 0.28,
                "target": "estancia_prolongada_a",
            }
        }
    }
    result = predict_patient(
        "A", {"edad": 52}, {"A": ProbabilityStub(0.34)}, metadata
    )
    assert result["label"] == "SÍ"
    assert result["prediction"] == 1
    assert result["probability"] == 0.34


def test_prediction_rejects_missing_variables() -> None:
    """Impide inferencias silenciosas con un contrato incompleto."""
    metadata = {
        "models": {
            "B": {
                "predictors": ["edad", "conducta"],
                "threshold": 0.26,
                "target": "estancia_prolongada_b",
            }
        }
    }
    try:
        predict_patient("B", {"edad": 52}, {"B": ProbabilityStub(0.3)}, metadata)
    except ValueError as error:
        assert "conducta" in str(error)
    else:
        raise AssertionError("La inferencia debía rechazar variables faltantes.")


def test_episode_predictors_derive_operational_context() -> None:
    """Calcula carga y grupo de cama sin entradas manuales del usuario."""
    context = pd.DataFrame(
        {
            "edad": [40, 60],
            "triage": ["triage_3", "triage_2"],
            "afiliacion": ["otras_eapb", "otras_eapb"],
            "fecha_ingreso": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
            "fecha_conducta": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 12:00"]),
            "fecha_salida": pd.to_datetime(["2026-01-01 15:00", "2026-01-01 18:00"]),
            "sala_observacion": ["norte_sala_a", "norte_sala_a"],
            "cama": ["A CAMA 1", "A CAMA 2"],
            "conducta": ["salida", "hospitalizacion"],
        }
    )
    predictors = build_episode_predictors(context, "B", 1)
    assert predictors["grupo_cama"] == "a_cama"
    assert predictors["pacientes_pendientes_de_salida"] == 1
    assert predictors["duracion_ingreso_calculada"] == 3.0


def test_new_patient_uses_observed_inputs_and_derives_load() -> None:
    """Mantiene interactivos datos observados y calcula internamente la carga."""
    context = pd.DataFrame(
        {
            "fecha_ingreso": pd.to_datetime(["2026-01-01 08:00"]),
            "fecha_conducta": pd.to_datetime(["2026-01-01 10:00"]),
            "fecha_salida": pd.to_datetime(["2026-01-01 15:00"]),
        }
    )
    patient = {
        "edad": 52,
        "triage": "triage_3",
        "afiliacion": "otras_eapb",
        "fecha_ingreso": "2026-01-01 09:00",
        "fecha_conducta": "2026-01-01 12:00",
        "conducta": "salida",
        "sala_observacion": "norte_sala_a",
        "cama": "A CAMA 2",
    }
    predictors = build_new_patient_predictors(context, "B", patient)
    assert predictors["duracion_ingreso_calculada"] == 3.0
    assert predictors["pacientes_pendientes_de_salida"] == 1
    assert predictors["grupo_cama"] == "a_cama"
