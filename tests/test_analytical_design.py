"""Pruebas de la selección analítica previa a la limpieza."""

import pandas as pd
import pytest

from src.analytical_design import materialize_analytical_design


def test_materialize_analytical_design_keeps_only_authorized_columns() -> None:
    """La vista debe contener la unión de predictores y controles."""
    source = pd.DataFrame({"edad": [20, 30], "conducta": ["A", "B"], "nota": [1, 2]})
    selection = pd.DataFrame(
        [
            {
                "columna": "edad",
                "modelo_a": "incluir",
                "modelo_b": "incluir",
                "rol": "predictora",
                "justificacion": "Disponible.",
            },
            {
                "columna": "conducta",
                "modelo_a": "excluir",
                "modelo_b": "incluir",
                "rol": "predictora",
                "justificacion": "Disponible para B.",
            },
            {
                "columna": "nota",
                "modelo_a": "excluir",
                "modelo_b": "excluir",
                "rol": "control_objetivo",
                "justificacion": "Control.",
            },
        ]
    )

    output, metrics = materialize_analytical_design(source, selection, ["nota"])

    assert output.columns.tolist() == ["edad", "conducta", "nota"]
    assert metrics["predictores_modelo_a"] == ["edad"]
    assert metrics["predictores_modelo_b"] == ["edad", "conducta"]


def test_materialize_analytical_design_rejects_missing_source_column() -> None:
    """Una selección desalineada debe producir un error explícito."""
    selection = pd.DataFrame(
        [
            {
                "columna": "inexistente",
                "modelo_a": "incluir",
                "modelo_b": "excluir",
                "rol": "predictora",
                "justificacion": "Prueba.",
            }
        ]
    )

    with pytest.raises(ValueError, match="No existen en la etapa 01"):
        materialize_analytical_design(pd.DataFrame({"edad": [20]}), selection, [])
