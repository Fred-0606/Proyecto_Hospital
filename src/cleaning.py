"""Aplica las reglas de limpieza y calidad de la etapa 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_io import (
    require_file,
    save_dataframe_as_csv,
    save_dataframe_as_parquet,
    save_json,
)
from src.paths import INTERIM_DATA_DIR, METRICS_DIR, TABLES_DIR

STAGE_02_SELECTED_PATH = INTERIM_DATA_DIR / "02_variables_seleccionadas.parquet"
STAGE_03_OUTPUT_PATH = INTERIM_DATA_DIR / "03_datos_para_analisis.parquet"
STAGE_03_NULLS_PATH = TABLES_DIR / "03_resumen_nulos.csv"
STAGE_03_TYPES_PATH = TABLES_DIR / "03_conversion_tipos.csv"
STAGE_03_TEMPORAL_PATH = TABLES_DIR / "03_auditoria_duraciones.csv"
STAGE_03_CATEGORIES_PATH = TABLES_DIR / "03_auditoria_categorias.csv"
STAGE_03_SPECIALTIES_PATH = TABLES_DIR / "03_vocabulario_especialidades.csv"
STAGE_03_METRICS_PATH = METRICS_DIR / "03_calidad_datos.json"
def encode_columns_for_variance_threshold(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Codifica temporalmente cada columna para evaluar su varianza con sklearn."""
    encoded = pd.DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        codes, _ = pd.factorize(dataframe[column], sort=True, use_na_sentinel=True)
        encoded[column] = pd.Series(codes, index=dataframe.index, dtype="float64").replace(
            -1,
            np.nan,
        )
    return encoded


def run_stage_03_quality(
    input_path: Path = STAGE_02_SELECTED_PATH,
    output_path: Path = STAGE_03_OUTPUT_PATH,
    dataframe: pd.DataFrame | None = None,
    temporal_audit: pd.DataFrame | None = None,
    categorical_audit: pd.DataFrame | None = None,
    specialty_vocabulary: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Convierte tipos y audita faltantes sin aplicar imputación estadística."""
    if dataframe is None:
        require_file(
            input_path,
            "Ejecute primero la etapa 02 para generar 02_variables_seleccionadas.parquet.",
        )
        source = pd.read_parquet(input_path)
        dataframe, type_audit = convert_selected_column_types(source)
    else:
        source = pd.read_parquet(input_path) if input_path.is_file() else dataframe
        type_audit = build_type_conversion_audit(source, dataframe)
    nulls = (
        dataframe.isna()
        .sum()
        .rename("valores_nulos")
        .rename_axis("columna")
        .reset_index()
    )
    nulls["porcentaje_nulos"] = (
        nulls["valores_nulos"].div(len(dataframe)).mul(100).round(4)
    )
    nulls = nulls.sort_values("valores_nulos", ascending=False).reset_index(drop=True)

    save_dataframe_as_parquet(dataframe, output_path)
    tables = (
        (nulls, STAGE_03_NULLS_PATH),
        (type_audit, STAGE_03_TYPES_PATH),
        (temporal_audit, STAGE_03_TEMPORAL_PATH),
        (categorical_audit, STAGE_03_CATEGORIES_PATH),
        (specialty_vocabulary, STAGE_03_SPECIALTIES_PATH),
    )
    for table, path in tables:
        if table is not None:
            save_dataframe_as_csv(table, path)
    metrics = {
        "filas": dataframe.shape[0],
        "columnas": dataframe.shape[1],
        "celdas_nulas": int(dataframe.isna().sum().sum()),
        "filas_con_algun_nulo": int(dataframe.isna().any(axis=1).sum()),
        "valores_convertidos_a_nulo": int(type_audit["nulos_nuevos"].sum()),
        "imputacion_deterministica_cama_aplicada": bool(
            dataframe["cama"].eq("cama_sin_registrar").any()
        ),
        "registros_cama_sin_registrar": int(
            dataframe["cama"].eq("cama_sin_registrar").sum()
        ),
        "imputacion_estadistica_aplicada": False,
    }
    if temporal_audit is not None:
        metrics["auditoria_duraciones"] = temporal_audit.to_dict(orient="records")
    if categorical_audit is not None:
        metrics["auditoria_categorias"] = categorical_audit.to_dict(orient="records")
    if specialty_vocabulary is not None:
        metrics["especialidades_homologadas"] = len(specialty_vocabulary)
    save_json(metrics, STAGE_03_METRICS_PATH)
    return {
        "dataframe": dataframe,
        "nulls": nulls,
        "type_audit": type_audit,
        "temporal_audit": temporal_audit,
        "categorical_audit": categorical_audit,
        "specialty_vocabulary": specialty_vocabulary,
        "metrics": metrics,
    }


def to_nullable_integer(series: pd.Series) -> pd.Series:
    """Convierte a Int64 únicamente edades enteras entre 0 y 120 años."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid_ages = numeric.mod(1).eq(0) & numeric.between(0, 120, inclusive="both")
    return numeric.where(valid_ages).astype("Int64")


def duration_to_hours(series: pd.Series) -> pd.Series:
    """Convierte duraciones reconocibles por pandas a horas decimales."""
    text = series.astype("string").str.strip()
    hours_and_minutes = text.str.fullmatch(r"[+-]?\d+:\d{1,2}", na=False)
    normalized = text.mask(hours_and_minutes, text + ":00")
    durations = pd.to_timedelta(normalized, errors="coerce")
    return durations.dt.total_seconds().div(3600).astype("float64")


def to_datetime_mixed(series: pd.Series) -> pd.Series:
    """Convierte fechas mixtas a datetime y transforma valores inválidos en NaT."""
    return pd.to_datetime(series, format="mixed", errors="coerce")


def convert_selected_column_types(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica los tipos finales acordados a edad, duraciones y fechas."""
    converted = dataframe.copy(deep=True)
    converted["edad"] = to_nullable_integer(converted["edad"])
    for column in ("tiempo_en_ingreso", "total_en_salida"):
        converted[column] = duration_to_hours(converted[column])
    for column in ("fecha_ingreso", "fecha_conducta", "fecha_salida"):
        converted[column] = to_datetime_mixed(converted[column])
    return converted, build_type_conversion_audit(dataframe, converted)


def impute_missing_bed_category(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Representa camas ausentes mediante una categoría explícita y determinística."""
    imputed = dataframe.copy(deep=True)
    imputed["cama"] = imputed["cama"].fillna("cama_sin_registrar")
    return imputed


def recalculate_stay_durations(
    dataframe: pd.DataFrame,
    tolerance_minutes: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea duraciones oficiales desde fechas y audita los valores originales."""
    reconciled = dataframe.copy(deep=True)
    definitions = (
        (
            "tiempo_en_ingreso",
            "duracion_ingreso_calculada",
            "fecha_ingreso",
            "fecha_conducta",
        ),
        (
            "total_en_salida",
            "duracion_salida_calculada",
            "fecha_conducta",
            "fecha_salida",
        ),
    )
    audit_rows = []

    for original_column, calculated_column, start_column, end_column in definitions:
        registered = reconciled[original_column]
        calculated = (
            reconciled[end_column] - reconciled[start_column]
        ).dt.total_seconds().div(3600)
        negative = calculated.lt(0)
        final_duration = calculated.mask(negative)
        comparable = registered.notna() & calculated.notna()
        discrepancy = registered.sub(calculated).abs()
        tolerance_hours = tolerance_minutes / 60

        audit_rows.append(
            {
                "duracion_original": original_column,
                "duracion_calculada": calculated_column,
                "formula": f"{end_column} - {start_column}",
                "nulos_registrados": int(registered.isna().sum()),
                "recuperados_desde_fechas": int(
                    (registered.isna() & final_duration.notna()).sum()
                ),
                "diferencias_mayores_un_minuto": int(
                    (comparable & discrepancy.gt(tolerance_hours)).sum()
                ),
                "diferencia_maxima_horas": (
                    float(discrepancy[comparable].max()) if comparable.any() else None
                ),
                "cambios_clasificacion_umbral_6h": int(
                    (
                        comparable
                        & registered.ge(6.0).ne(calculated.ge(6.0))
                    ).sum()
                ),
                "duraciones_negativas": int(negative.sum()),
                "nulos_finales": int(final_duration.isna().sum()),
            }
        )
        reconciled[calculated_column] = final_duration.astype("float64")

    return reconciled, pd.DataFrame(audit_rows)


def build_type_conversion_audit(
    source: pd.DataFrame,
    converted: pd.DataFrame,
) -> pd.DataFrame:
    """Compara tipos y cuenta nulos creados por cada conversión."""
    def display_type(dtype: object) -> str:
        """Unifica las representaciones textuales str y string."""
        dtype_name = str(dtype)
        return "string" if dtype_name in {"str", "string"} else dtype_name

    rows = []
    for column in converted.columns:
        is_derived = column not in source.columns
        original_nulls = 0 if is_derived else int(source[column].isna().sum())
        final_nulls = int(converted[column].isna().sum())
        rows.append(
            {
                "columna": column,
                "tipo_original": (
                    "creada_en_etapa_03"
                    if is_derived
                    else display_type(source[column].dtype)
                ),
                "tipo_final": display_type(converted[column].dtype),
                "nulos_originales": original_nulls,
                "nulos_finales": final_nulls,
                "nulos_nuevos": 0 if is_derived else max(final_nulls - original_nulls, 0),
            }
        )
    return pd.DataFrame(rows)
