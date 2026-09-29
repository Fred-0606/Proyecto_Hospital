"""Define y materializa las vistas analíticas de los modelos A y B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_io import (
    require_file,
    save_dataframe_as_csv,
    save_dataframe_as_parquet,
    save_json,
)
from src.paths import INTERIM_DATA_DIR, METRICS_DIR, TABLES_DIR

STAGE_01_INPUT_PATH = INTERIM_DATA_DIR / "01_censo_identificado.parquet"
STAGE_02_OUTPUT_PATH = INTERIM_DATA_DIR / "02_variables_seleccionadas.parquet"
STAGE_02_AUDIT_PATH = TABLES_DIR / "02_matriz_seleccion_variables.csv"
STAGE_02_METRICS_PATH = METRICS_DIR / "02_diseno_analitico.json"


def materialize_analytical_design(
    source: pd.DataFrame,
    selection: pd.DataFrame,
    control_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Proyecta las columnas autorizadas y devuelve métricas auditables."""
    required_fields = {"columna", "modelo_a", "modelo_b", "rol", "justificacion"}
    missing_fields = required_fields.difference(selection.columns)
    if missing_fields:
        raise ValueError(f"Faltan campos en la matriz de selección: {missing_fields}")

    selected = selection.loc[
        selection["modelo_a"].eq("incluir")
        | selection["modelo_b"].eq("incluir")
        | selection["columna"].isin(control_columns),
        "columna",
    ].tolist()
    missing_columns = sorted(set(selected).difference(source.columns))
    if missing_columns:
        raise ValueError(f"No existen en la etapa 01: {missing_columns}")

    output = source.loc[:, selected].copy(deep=True)
    metrics: dict[str, object] = {
        "filas": output.shape[0],
        "columnas_iniciales": source.shape[1],
        "columnas_para_calidad": selected,
        "predictores_modelo_a": selection.loc[
            selection["modelo_a"].eq("incluir") & selection["rol"].eq("predictora"),
            "columna",
        ].tolist(),
        "predictores_modelo_b": selection.loc[
            selection["modelo_b"].eq("incluir") & selection["rol"].eq("predictora"),
            "columna",
        ].tolist(),
        "controles_calidad": control_columns,
    }
    return output, metrics


def run_stage_02(
    selection: pd.DataFrame,
    control_columns: list[str],
    input_path: Path = STAGE_01_INPUT_PATH,
    output_path: Path = STAGE_02_OUTPUT_PATH,
) -> dict[str, object]:
    """Ejecuta la selección analítica y guarda su vista y auditoría."""
    require_file(
        input_path,
        "Ejecute primero la etapa 01 para generar 01_censo_identificado.parquet.",
    )
    source = pd.read_parquet(input_path)
    output, metrics = materialize_analytical_design(
        source,
        selection,
        control_columns,
    )
    save_dataframe_as_parquet(output, output_path)
    save_dataframe_as_csv(selection, STAGE_02_AUDIT_PATH)
    save_json(metrics, STAGE_02_METRICS_PATH)
    return {"dataframe": output, "selection": selection, "metrics": metrics}
