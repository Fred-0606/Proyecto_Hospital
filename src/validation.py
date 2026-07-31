"""Ejecuta validaciones informativas de calidad sin alterar los datos de CENSO."""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from src.data_io import (
    build_ingestion_metadata,
    get_identified_parquet_path,
    get_source_excel_path,
    get_excel_sheet_names,
    normalize_dataframe_columns,
    identify_mixed_object_columns,
    project_relative_path,
    read_censo_sheet,
    save_dataframe_as_parquet,
    save_ingestion_metadata,
    save_json,
)
from src.paths import FIGURES_DIR, METRICS_DIR, TABLES_DIR


TEXT_NULL_TOKENS = {
    "NA",
    "N/A",
    "NULL",
    "NONE",
    "SIN DATO",
    "NO APLICA",
    "VACIO",
    "-",
    "--",
    ".",
    "NO REGISTRA",
}
IDENTIFIER_TERMS = {
    "id",
    "identificacion",
    "documento",
    "episodio",
    "admision",
    "ingreso",
    "historia",
    "paciente",
    "consecutivo",
}
DATE_TERMS = {
    "fecha",
    "hora",
    "ingreso",
    "egreso",
    "salida",
    "triage",
    "atencion",
    "admision",
    "hospitalizacion",
}
DATE_NAME_EXCLUSIONS = {
    "clase",
    "estado",
    "meta",
    "orden",
    "tiempo",
    "total",
    "hrs",
    "dias",
}


def _percentage(numerator: int | float, denominator: int | float) -> float:
    """Calcula un porcentaje seguro."""
    return float(numerator / denominator * 100) if denominator else 0.0


def _fold_text(value: object) -> str:
    """Normaliza texto para comparaciones sin modificar el valor fuente."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _category_key(value: object) -> str:
    """Crea una clave comparable ignorando espacios, tildes y puntuación."""
    text = _fold_text(value)
    return "".join(character for character in text if character not in string.punctuation)


def _write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Guarda una evidencia tabular en CSV UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = (
        dataframe
        if len(dataframe.columns) > 0
        else pd.DataFrame(columns=["sin_hallazgos"])
    )
    table.to_csv(path, index=False, encoding="utf-8")
    return path


def calculate_dimensions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calcula dimensiones, celdas informadas, nulos y completitud global."""
    rows, columns = dataframe.shape
    total_cells = rows * columns
    null_cells = int(dataframe.isna().sum().sum())
    informed_cells = total_cells - null_cells
    return pd.DataFrame(
        [
            {
                "filas": rows,
                "columnas": columns,
                "celdas_totales": total_cells,
                "celdas_con_informacion": informed_cells,
                "celdas_nulas": null_cells,
                "completitud_porcentaje": _percentage(informed_cells, total_cells),
            }
        ]
    )


def build_column_summary(
    dataframe: pd.DataFrame,
    original_by_normalized: dict[str, str],
) -> pd.DataFrame:
    """Resume nombres, tipos, valores presentes y unicidad por columna."""
    rows = []
    for position, column in enumerate(dataframe.columns, start=1):
        series = dataframe[column]
        rows.append(
            {
                "posicion": position,
                "nombre_original": original_by_normalized[column],
                "nombre_normalizado": column,
                "tipo_inferido_pandas": str(series.dtype),
                "valores_no_nulos": int(series.notna().sum()),
                "valores_unicos": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _python_type_names(series: pd.Series, sample_size: int | None = None) -> list[str]:
    """Obtiene tipos Python reales presentes en valores no nulos."""
    values = series.dropna()
    if sample_size is not None:
        values = values.head(sample_size)
    return sorted({type(value).__name__ for value in values})


def analyze_data_types(dataframe: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Clasifica dtypes y detecta columnas con tipos Python mixtos."""
    records = []
    for column in dataframe.columns:
        series = dataframe[column]
        all_types = _python_type_names(series)
        sample_types = _python_type_names(series, sample_size)
        records.append(
            {
                "columna": column,
                "dtype_pandas": str(series.dtype),
                "es_numerica": ptypes.is_numeric_dtype(series),
                "es_texto": ptypes.is_string_dtype(series),
                "es_booleana": ptypes.is_bool_dtype(series),
                "es_fecha_reconocida": ptypes.is_datetime64_any_dtype(series),
                "es_object": ptypes.is_object_dtype(series),
                "tipos_python_muestra": ", ".join(sample_types),
                "tipos_python_no_nulos": ", ".join(all_types),
                "contiene_tipos_mixtos": len(all_types) > 1,
            }
        )
    return pd.DataFrame(records)


def analyze_nulls(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Calcula nulos por columna y la distribución de nulos por fila."""
    row_count = len(dataframe)
    nulls_by_column = []
    for column in dataframe.columns:
        null_count = int(dataframe[column].isna().sum())
        non_null_count = row_count - null_count
        nulls_by_column.append(
            {
                "columna": column,
                "nulos": null_count,
                "nulos_porcentaje": _percentage(null_count, row_count),
                "no_nulos": non_null_count,
                "completitud_porcentaje": _percentage(non_null_count, row_count),
            }
        )
    row_null_counts = dataframe.isna().sum(axis=1)
    distribution = (
        row_null_counts.value_counts()
        .sort_index()
        .rename_axis("nulos_en_fila")
        .reset_index(name="cantidad_filas")
    )
    distribution["porcentaje_filas"] = distribution["cantidad_filas"].map(
        lambda count: _percentage(int(count), row_count)
    )
    summary = {
        "filas_completamente_vacias": int(dataframe.isna().all(axis=1).sum()),
        "filas_con_al_menos_un_nulo": int(dataframe.isna().any(axis=1).sum()),
        "filas_completas": int(dataframe.notna().all(axis=1).sum()),
    }
    return pd.DataFrame(nulls_by_column), distribution, summary


def plot_missing_values(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Guarda un gráfico de porcentaje de nulos por columna con colores por defecto."""
    percentages = dataframe.isna().mean().mul(100).sort_values(ascending=False)
    figure_width = max(8.0, min(24.0, len(percentages) * 0.35))
    figure, axis = plt.subplots(figsize=(figure_width, 6))
    percentages.plot(kind="bar", ax=axis)
    axis.set_title("Porcentaje de valores nulos por columna")
    axis.set_xlabel("Columna")
    axis.set_ylabel("Porcentaje de nulos")
    axis.tick_params(axis="x", labelrotation=90)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def analyze_exact_duplicates(
    dataframe: pd.DataFrame,
    example_limit: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resume duplicados exactos y conserva ejemplos con índices fuente."""
    duplicated_mask = dataframe.duplicated(keep=False)
    duplicated_rows = dataframe.loc[duplicated_mask]
    duplicate_count = int(duplicated_mask.sum())
    if duplicated_rows.empty:
        group_sizes = pd.Series(dtype="int64")
    else:
        group_sizes = duplicated_rows.groupby(
            list(dataframe.columns), dropna=False, sort=False
        ).size()
    summary = {
        "filas_duplicadas_exactas": duplicate_count,
        "filas_duplicadas_porcentaje": _percentage(duplicate_count, len(dataframe)),
        "grupos_duplicados": int(len(group_sizes)),
        "tamano_maximo_grupo": int(group_sizes.max()) if not group_sizes.empty else 0,
        "indices_duplicados": [int(index) for index in dataframe.index[duplicated_mask]],
    }
    summary_table = pd.DataFrame(
        [{key: value if key != "indices_duplicados" else json_list(value) for key, value in summary.items()}]
    )
    examples = duplicated_rows.head(example_limit).copy()
    examples.insert(0, "indice_original", examples.index)
    return summary_table, examples.reset_index(drop=True), summary


def json_list(values: Iterable[Any]) -> str:
    """Representa una lista de forma estable para evidencias CSV."""
    return "[" + ", ".join(str(value) for value in values) + "]"


def detect_identifier_candidates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Detecta candidatos a identificador mediante nombre, unicidad y patrón."""
    records = []
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna().astype(str)
        non_null_count = len(non_null)
        unique_count = int(non_null.nunique())
        uniqueness = _percentage(unique_count, non_null_count)
        normalized_name = _fold_text(column).replace(" ", "_")
        name_match = any(term in normalized_name for term in IDENTIFIER_TERMS)
        numeric_pattern = float(non_null.str.fullmatch(r"\d+").mean() * 100) if non_null_count else 0.0
        alphanumeric_pattern = (
            float(non_null.str.fullmatch(r"[A-Za-z0-9._-]+").mean() * 100)
            if non_null_count
            else 0.0
        )
        candidate = name_match or (
            non_null_count > 0
            and uniqueness >= 95
            and max(numeric_pattern, alphanumeric_pattern) >= 80
        )
        if candidate:
            lengths = non_null.str.len()
            records.append(
                {
                    "columna": column,
                    "coincidencia_nombre": name_match,
                    "unicidad_porcentaje": uniqueness,
                    "valores_repetidos": non_null_count - unique_count,
                    "nulos": int(series.isna().sum()),
                    "longitud_minima": int(lengths.min()) if not lengths.empty else None,
                    "longitud_maxima": int(lengths.max()) if not lengths.empty else None,
                    "longitud_mediana": float(lengths.median()) if not lengths.empty else None,
                    "patron_numerico_porcentaje": numeric_pattern,
                    "patron_alfanumerico_porcentaje": alphanumeric_pattern,
                }
            )
    return pd.DataFrame(records)


def analyze_empty_constant_columns(
    dataframe: pd.DataFrame,
    low_variability_max_unique: int = 5,
) -> pd.DataFrame:
    """Identifica columnas vacías, constantes y dominadas por un valor."""
    records = []
    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        top_share = (
            float(non_null.value_counts(dropna=False, normalize=True).iloc[0] * 100)
            if not non_null.empty
            else 0.0
        )
        records.append(
            {
                "columna": column,
                "completamente_vacia": non_null.empty,
                "unico_valor_no_nulo": unique_count == 1,
                "baja_variabilidad": 1 < unique_count <= low_variability_max_unique,
                "mas_95_porcentaje_mismo_valor": top_share > 95,
                "valores_unicos_no_nulos": unique_count,
                "valor_dominante_porcentaje": top_share,
            }
        )
    return pd.DataFrame(records)


def analyze_text_problems(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detecta problemas de espacios, capitalización, caracteres y nulos textuales."""
    problem_records = []
    null_token_records = []
    for column in dataframe.columns:
        if not (ptypes.is_object_dtype(dataframe[column]) or ptypes.is_string_dtype(dataframe[column])):
            continue
        values = dataframe[column].dropna()
        text = values.astype(str)
        folded = text.map(_fold_text)
        case_variants = int(
            text.groupby(folded).nunique().gt(1).sum()
        )
        non_printable_count = sum(
            any(not character.isprintable() for character in value)
            for value in text
        )
        problem_records.append(
            {
                "columna": column,
                "espacios_iniciales": int(text.str.match(r"^\s+").sum()),
                "espacios_finales": int(text.str.match(r".*\s+$").sum()),
                "dobles_espacios": int(text.str.contains(r"\s{2,}", regex=True).sum()),
                "grupos_con_diferencias_mayusculas": case_variants,
                "cadenas_vacias": int(text.eq("").sum()),
                "solo_espacios": int(text.str.fullmatch(r"\s+").sum()),
                "caracteres_no_imprimibles": int(non_printable_count),
            }
        )
        normalized_tokens = text.map(lambda value: _fold_text(value).upper())
        token_counts = normalized_tokens[normalized_tokens.isin(TEXT_NULL_TOKENS)].value_counts()
        for token, count in token_counts.items():
            null_token_records.append(
                {"columna": column, "valor_normalizado": token, "cantidad": int(count)}
            )
    return pd.DataFrame(problem_records), pd.DataFrame(null_token_records)


def analyze_categorical_columns(
    dataframe: pd.DataFrame,
    max_categories: int = 50,
    max_cardinality_percentage: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume categorías razonables y propone equivalencias sin consolidarlas."""
    summaries = []
    frequencies = []
    equivalences = []
    row_count = len(dataframe)
    for column in dataframe.columns:
        series = dataframe[column]
        if not (ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)):
            continue
        non_null = series.dropna().astype(str)
        categories = int(non_null.nunique())
        cardinality = _percentage(categories, len(non_null))
        if categories > max_categories and cardinality > max_cardinality_percentage:
            continue
        counts = non_null.value_counts()
        top_value = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        summaries.append(
            {
                "columna": column,
                "numero_categorias": categories,
                "cardinalidad_porcentaje": cardinality,
                "categoria_mas_frecuente": top_value,
                "frecuencia_categoria_mas_frecuente": top_count,
                "categoria_mas_frecuente_porcentaje": _percentage(top_count, len(non_null)),
                "registros_totales": row_count,
            }
        )
        for rank, (category, count) in enumerate(counts.head(10).items(), start=1):
            frequencies.append(
                {
                    "columna": column,
                    "posicion": rank,
                    "categoria": category,
                    "frecuencia": int(count),
                    "porcentaje": _percentage(int(count), len(non_null)),
                }
            )
        grouped: dict[str, set[str]] = {}
        for category in counts.index:
            grouped.setdefault(_category_key(category), set()).add(category)
        for key, variants in grouped.items():
            if key and len(variants) > 1:
                equivalences.append(
                    {
                        "columna": column,
                        "clave_comparable": key,
                        "categorias_originales": " | ".join(sorted(variants)),
                        "cantidad_variantes": len(variants),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(frequencies), pd.DataFrame(equivalences)


def analyze_numeric_columns(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula estadísticas numéricas e identifica extremos mediante IQR."""
    summaries = []
    outlier_records = []
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        quantiles = finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        q1 = float(quantiles.get(0.25, np.nan))
        q3 = float(quantiles.get(0.75, np.nan))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = series.lt(lower) | series.gt(upper)
        summaries.append(
            {
                "columna": column,
                "cantidad": int(series.notna().sum()),
                "nulos": int(series.isna().sum()),
                "media": float(finite.mean()) if not finite.empty else np.nan,
                "mediana": float(finite.median()) if not finite.empty else np.nan,
                "desviacion_estandar": float(finite.std()) if len(finite) > 1 else np.nan,
                "minimo": float(finite.min()) if not finite.empty else np.nan,
                "maximo": float(finite.max()) if not finite.empty else np.nan,
                "percentil_1": quantiles.get(0.01, np.nan),
                "percentil_5": quantiles.get(0.05, np.nan),
                "percentil_25": quantiles.get(0.25, np.nan),
                "percentil_50": quantiles.get(0.5, np.nan),
                "percentil_75": quantiles.get(0.75, np.nan),
                "percentil_95": quantiles.get(0.95, np.nan),
                "percentil_99": quantiles.get(0.99, np.nan),
                "rango_intercuartilico": iqr,
                "ceros": int(series.eq(0).sum()),
                "valores_negativos": int(series.lt(0).sum()),
                "infinitos": int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()),
                "valores_extremos_iqr": int(outlier_mask.sum()),
            }
        )
        for index, value in series[outlier_mask].items():
            outlier_records.append(
                {
                    "columna": column,
                    "indice_original": index,
                    "valor": value,
                    "limite_inferior": lower,
                    "limite_superior": upper,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(outlier_records)


def _convert_dates(series: pd.Series) -> pd.Series:
    """Convierte prudentemente una serie a fecha sin modificar el origen."""
    if ptypes.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)
    return pd.to_datetime(
        series.astype("string"),
        errors="coerce",
        format="mixed",
        utc=True,
    )


def detect_date_candidates(
    dataframe: pd.DataFrame,
    earliest_date: str = "1900-01-01",
    content_threshold: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Detecta candidatas por nombre o contenido, evitando numéricas sin indicios."""
    candidates = []
    results = []
    failures = []
    converted_by_column: dict[str, pd.Series] = {}
    now = pd.Timestamp.now(tz="UTC")
    earliest = pd.Timestamp(earliest_date, tz="UTC")
    for column in dataframe.columns:
        series = dataframe[column]
        folded_name = _fold_text(column).replace("_", " ")
        name_tokens = set(folded_name.split())
        has_explicit_date_word = bool({"fecha", "hora"} & name_tokens)
        has_temporal_name = bool(DATE_TERMS & name_tokens)
        has_exclusion = bool(DATE_NAME_EXCLUSIONS & name_tokens)
        name_match = has_explicit_date_word or (has_temporal_name and not has_exclusion)
        recognized_datetime = ptypes.is_datetime64_any_dtype(series)
        eligible_content = not ptypes.is_numeric_dtype(series)
        if not (name_match or eligible_content or recognized_datetime):
            continue
        non_null = int(series.notna().sum())
        if eligible_content and non_null:
            non_null_values = series.dropna()
            string_values = non_null_values.astype(str)
            date_pattern_share = float(
                string_values.str.contains(
                    r"(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(?:\d{1,2}:\d{2})",
                    regex=True,
                ).mean()
                * 100
            )
            python_date_share = float(
                non_null_values.map(
                    lambda value: isinstance(
                        value,
                        (pd.Timestamp, datetime, np.datetime64),
                    )
                ).mean()
                * 100
            )
        else:
            date_pattern_share = 0.0
            python_date_share = 0.0
        content_hint = max(date_pattern_share, python_date_share) >= content_threshold
        if not (name_match or recognized_datetime or content_hint):
            continue
        converted = _convert_dates(series)
        convertible = int(converted.notna().sum())
        conversion_rate = _percentage(convertible, non_null)
        content_match = content_hint and conversion_rate >= content_threshold
        candidates.append(
            {
                "columna": column,
                "detectada_por_nombre": name_match,
                "detectada_por_contenido": content_match,
                "dtype_original": str(series.dtype),
                "patron_fecha_hora_porcentaje": date_pattern_share,
            }
        )
        converted_by_column[column] = converted
        failed_mask = series.notna() & converted.isna()
        valid = converted.dropna()
        results.append(
            {
                "columna": column,
                "no_nulos_originales": non_null,
                "fechas_convertibles": convertible,
                "fechas_no_convertibles": int(failed_mask.sum()),
                "conversion_porcentaje": conversion_rate,
                "fecha_minima": valid.min() if not valid.empty else pd.NaT,
                "fecha_maxima": valid.max() if not valid.empty else pd.NaT,
                "fechas_futuras": int(converted.gt(now).sum()),
                "fechas_anteriores_limite": int(converted.lt(earliest).sum()),
                "limite_inferior_configurado": earliest.date().isoformat(),
            }
        )
        for index, value in series[failed_mask].head(20).items():
            failures.append(
                {"columna": column, "indice_original": index, "valor_original": value}
            )
    return (
        pd.DataFrame(candidates),
        pd.DataFrame(results),
        pd.DataFrame(failures),
        converted_by_column,
    )


TEMPORAL_RELATIONS = (
    (("ingreso", "admision", "triage", "inicio", "solicitud", "conducta"), ("salida", "egreso", "fin", "realizacion", "conducta")),
)


def analyze_temporal_consistency(
    converted_by_column: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa pares temporales plausibles existentes, sin inventar columnas."""
    summaries = []
    inconsistencies = []
    columns = list(converted_by_column)
    seen: set[tuple[str, str]] = set()
    for start_column in columns:
        start_name = _fold_text(start_column)
        for end_column in columns:
            if start_column == end_column:
                continue
            end_name = _fold_text(end_column)
            plausible = any(
                any(term in start_name for term in start_terms)
                and any(term in end_name for term in end_terms)
                for start_terms, end_terms in TEMPORAL_RELATIONS
            )
            pair = (start_column, end_column)
            if not plausible or pair in seen:
                continue
            seen.add(pair)
            start = converted_by_column[start_column]
            end = converted_by_column[end_column]
            valid_mask = start.notna() & end.notna()
            differences = end[valid_mask] - start[valid_mask]
            negative_mask = differences < pd.Timedelta(0)
            zero_mask = differences == pd.Timedelta(0)
            summaries.append(
                {
                    "columna_inicial": start_column,
                    "columna_final": end_column,
                    "registros_ambas_fechas_validas": int(valid_mask.sum()),
                    "fecha_final_anterior": int(negative_mask.sum()),
                    "diferencia_cero": int(zero_mask.sum()),
                    "diferencia_minima_horas": differences.min().total_seconds() / 3600 if not differences.empty else np.nan,
                    "diferencia_maxima_horas": differences.max().total_seconds() / 3600 if not differences.empty else np.nan,
                    "diferencia_mediana_horas": differences.median().total_seconds() / 3600 if not differences.empty else np.nan,
                }
            )
            for index in differences[negative_mask].index:
                inconsistencies.append(
                    {
                        "columna_inicial": start_column,
                        "columna_final": end_column,
                        "indice_original": index,
                        "fecha_inicial": start.loc[index],
                        "fecha_final": end.loc[index],
                        "diferencia_horas": differences.loc[index].total_seconds() / 3600,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(inconsistencies)


VARIABLE_GROUPS = {
    "identificacion": IDENTIFIER_TERMS,
    "demograficas": {"edad", "sexo", "genero", "residencia", "municipio"},
    "clinicas": {"antecedente", "sintoma", "signo", "clinico"},
    "triage": {"triage"},
    "diagnostico": {"diagnostico", "cie"},
    "especialidad": {"especialidad", "servicio"},
    "administrativas": {"eps", "asegurador", "autorizacion", "administrativo"},
    "temporales": DATE_TERMS,
    "ubicacion_o_cama": {"ubicacion", "cama", "sala", "piso"},
    "ocupacion": {"ocupacion", "capacidad"},
    "conducta": {"conducta", "destino"},
    "salida": {"salida", "egreso", "alta"},
}


def classify_variables(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Propone grupos de variables a partir de nombres y dtypes para revisión."""
    records = []
    for column in dataframe.columns:
        folded_name = _fold_text(column)
        matches = [
            group
            for group, terms in VARIABLE_GROUPS.items()
            if any(term in folded_name for term in terms)
        ]
        records.append(
            {
                "columna": column,
                "grupo_propuesto": matches[0] if matches else "otras",
                "grupos_alternativos": " | ".join(matches[1:]),
                "dtype_pandas": str(dataframe[column].dtype),
                "criterio": "coincidencia programada por nombre y tipo",
                "requiere_revision_humana": True,
            }
        )
    return pd.DataFrame(records)


def analyze_memory(
    dataframe: pd.DataFrame,
    source_path: Path,
    parquet_path: Path,
) -> pd.DataFrame:
    """Calcula memoria por columna y tamaños de archivos de origen y salida."""
    memory = dataframe.memory_usage(index=True, deep=True)
    source_size = source_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    compression_ratio = source_size / parquet_size if parquet_size else np.nan
    rows = [
        {
            "elemento": "dataframe_total",
            "columna": None,
            "memoria_bytes": int(memory.sum()),
            "archivo_excel_bytes": source_size,
            "archivo_parquet_bytes": parquet_size,
            "relacion_compresion_excel_parquet": compression_ratio,
        }
    ]
    rows.extend(
        {
            "elemento": "columna",
            "columna": column,
            "memoria_bytes": int(value),
            "archivo_excel_bytes": np.nan,
            "archivo_parquet_bytes": np.nan,
            "relacion_compresion_excel_parquet": np.nan,
        }
        for column, value in memory.items()
    )
    return pd.DataFrame(rows)


def _summary_row(
    validation: str,
    warnings: int,
    affected: int,
    total: int,
    evidence: Path,
    observation_ok: str,
    observation_warning: str,
    review: bool = False,
) -> dict[str, Any]:
    """Construye una fila de resumen con reglas explícitas."""
    if warnings == 0:
        status = "OK"
        observation = observation_ok
    else:
        status = "REVISIÓN REQUERIDA" if review else "ADVERTENCIA"
        observation = observation_warning.format(cantidad=affected)
    return {
        "validacion": validation,
        "estado": status,
        "cantidad_advertencias": warnings,
        "registros_afectados": affected,
        "porcentaje_afectado": _percentage(affected, total),
        "archivo_evidencia": project_relative_path(evidence),
        "observacion_automatica": observation,
    }


def build_validation_summary(
    dataframe: pd.DataFrame,
    *,
    dimensions: pd.DataFrame,
    duplicate_summary: dict[str, Any],
    identifiers: pd.DataFrame,
    empty_constants: pd.DataFrame,
    text_problems: pd.DataFrame,
    textual_nulls: pd.DataFrame,
    date_results: pd.DataFrame,
    temporal_inconsistencies: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida estados y observaciones basadas únicamente en condiciones."""
    total_rows = len(dataframe)
    total_nulls = int(dimensions.iloc[0]["celdas_nulas"])
    duplicate_count = int(duplicate_summary["filas_duplicadas_exactas"])
    constant_count = int(
        (empty_constants["completamente_vacia"] | empty_constants["unico_valor_no_nulo"]).sum()
    )
    text_issue_count = int(
        text_problems.select_dtypes(include=[np.number]).sum().sum()
    ) if not text_problems.empty else 0
    textual_null_count = int(textual_nulls["cantidad"].sum()) if not textual_nulls.empty else 0
    date_failure_count = int(date_results["fechas_no_convertibles"].sum()) if not date_results.empty else 0
    temporal_count = len(temporal_inconsistencies)
    return pd.DataFrame(
        [
            _summary_row(
                "Valores nulos",
                int(total_nulls > 0),
                total_nulls,
                int(dimensions.iloc[0]["celdas_totales"]),
                TABLES_DIR / "01_valores_nulos.csv",
                "No se detectaron valores nulos.",
                "Se detectaron {cantidad} celdas nulas.",
            ),
            _summary_row(
                "Duplicados exactos",
                int(duplicate_count > 0),
                duplicate_count,
                total_rows,
                TABLES_DIR / "01_resumen_duplicados.csv",
                "No se detectaron filas duplicadas exactas.",
                "Se detectaron {cantidad} filas duplicadas; requieren revisión en la etapa 02.",
                review=True,
            ),
            _summary_row(
                "Identificadores candidatos",
                int(not identifiers.empty),
                len(identifiers),
                len(dataframe.columns),
                TABLES_DIR / "01_identificadores_candidatos.csv",
                "No se detectaron candidatos mediante las reglas configuradas.",
                "Se propusieron {cantidad} columnas candidatas para revisión.",
            ),
            _summary_row(
                "Columnas vacías o constantes",
                int(constant_count > 0),
                constant_count,
                len(dataframe.columns),
                TABLES_DIR / "01_columnas_vacias_constantes.csv",
                "No se detectaron columnas vacías o constantes.",
                "Se detectaron {cantidad} columnas vacías o constantes.",
                review=True,
            ),
            _summary_row(
                "Problemas de texto",
                int((text_issue_count + textual_null_count) > 0),
                text_issue_count + textual_null_count,
                max(1, int(dataframe.size)),
                TABLES_DIR / "01_problemas_texto.csv",
                "No se detectaron problemas de texto mediante las reglas configuradas.",
                "Se detectaron {cantidad} incidencias de texto para revisión.",
            ),
            _summary_row(
                "Conversión preliminar de fechas",
                int(date_failure_count > 0),
                date_failure_count,
                total_rows,
                TABLES_DIR / "01_resultado_conversion_fechas.csv",
                "No se detectaron valores no convertibles en las candidatas.",
                "Se detectaron {cantidad} valores no convertibles en candidatas a fecha.",
            ),
            _summary_row(
                "Consistencia temporal",
                int(temporal_count > 0),
                temporal_count,
                total_rows,
                TABLES_DIR / "01_inconsistencias_temporales.csv",
                "No se detectaron fechas finales anteriores a las iniciales.",
                "Se detectaron {cantidad} inconsistencias temporales preliminares.",
                review=True,
            ),
        ]
    )


def _empty_with_columns(columns: list[str]) -> pd.DataFrame:
    """Crea una tabla vacía con esquema estable para evidencias."""
    return pd.DataFrame(columns=columns)


def run_stage_01(earliest_date: str = "1900-01-01") -> dict[str, Any]:
    """Ejecuta la ingesta informativa completa y guarda todas las evidencias."""
    source_path = get_source_excel_path()
    sheet_names = get_excel_sheet_names(source_path)
    original_dataframe = read_censo_sheet(source_path)
    preserved_copy = original_dataframe.copy(deep=True)
    dataframe, original_by_normalized = normalize_dataframe_columns(original_dataframe)
    if not original_dataframe.equals(preserved_copy):
        raise RuntimeError("La copia original cambió durante la etapa de identificación.")

    parquet_string_columns = identify_mixed_object_columns(dataframe)
    parquet_path = save_dataframe_as_parquet(dataframe, get_identified_parquet_path())
    evidence_paths: list[Path] = [parquet_path]

    dimensions = calculate_dimensions(dataframe)
    columns = build_column_summary(dataframe, original_by_normalized)
    data_types = analyze_data_types(dataframe)
    nulls, nulls_per_row, null_row_summary = analyze_nulls(dataframe)
    duplicate_table, duplicate_examples, duplicate_summary = analyze_exact_duplicates(dataframe)
    identifiers = detect_identifier_candidates(dataframe)
    empty_constants = analyze_empty_constant_columns(dataframe)
    text_problems, textual_nulls = analyze_text_problems(dataframe)
    category_summary, category_frequencies, category_equivalences = analyze_categorical_columns(dataframe)
    numeric_summary, numeric_outliers = analyze_numeric_columns(dataframe)
    date_candidates, date_results, date_failures, converted_dates = detect_date_candidates(
        dataframe, earliest_date=earliest_date
    )
    temporal_summary, temporal_inconsistencies = analyze_temporal_consistency(converted_dates)
    classification = classify_variables(dataframe)
    memory = analyze_memory(dataframe, source_path, parquet_path)
    validation_summary = build_validation_summary(
        dataframe,
        dimensions=dimensions,
        duplicate_summary=duplicate_summary,
        identifiers=identifiers,
        empty_constants=empty_constants,
        text_problems=text_problems,
        textual_nulls=textual_nulls,
        date_results=date_results,
        temporal_inconsistencies=temporal_inconsistencies,
    )

    tables: dict[str, pd.DataFrame] = {
        "01_dimensiones.csv": dimensions,
        "01_columnas.csv": columns,
        "01_tipos_datos.csv": data_types,
        "01_valores_nulos.csv": nulls,
        "01_nulos_por_fila.csv": nulls_per_row,
        "01_resumen_duplicados.csv": duplicate_table,
        "01_ejemplos_duplicados.csv": duplicate_examples,
        "01_identificadores_candidatos.csv": identifiers,
        "01_columnas_vacias_constantes.csv": empty_constants,
        "01_problemas_texto.csv": text_problems,
        "01_nulos_representados_como_texto.csv": textual_nulls,
        "01_resumen_categorias.csv": category_summary,
        "01_frecuencias_categorias.csv": category_frequencies,
        "01_posibles_categorias_equivalentes.csv": category_equivalences,
        "01_resumen_numericas.csv": numeric_summary,
        "01_valores_extremos_iqr.csv": numeric_outliers,
        "01_columnas_fecha_candidatas.csv": date_candidates,
        "01_resultado_conversion_fechas.csv": date_results,
        "01_fechas_no_convertibles.csv": date_failures,
        "01_consistencia_temporal.csv": temporal_summary,
        "01_inconsistencias_temporales.csv": temporal_inconsistencies,
        "01_clasificacion_preliminar_variables.csv": classification,
        "01_memoria_dataframe.csv": memory,
        "01_resumen_validaciones.csv": validation_summary,
    }
    for filename, table in tables.items():
        evidence_paths.append(_write_csv(table, TABLES_DIR / filename))

    figure_path = plot_missing_values(dataframe, FIGURES_DIR / "01_valores_nulos.png")
    evidence_paths.append(figure_path)

    summary_json_path = save_json(
        {
            "fecha_hora_ejecucion": datetime.now().astimezone().isoformat(),
            "resumen_filas": validation_summary.to_dict(orient="records"),
            "filas_completamente_vacias": null_row_summary["filas_completamente_vacias"],
            "filas_con_al_menos_un_nulo": null_row_summary["filas_con_al_menos_un_nulo"],
            "filas_completas": null_row_summary["filas_completas"],
        },
        METRICS_DIR / "01_resumen_validaciones.json",
    )
    evidence_paths.append(summary_json_path)

    metadata = build_ingestion_metadata(
        source_path=source_path,
        sheet_names=sheet_names,
        dataframe=dataframe,
        parquet_path=parquet_path,
        evidence_paths=evidence_paths
        + [METRICS_DIR / "01_metadatos_ingesta.json"],
        parquet_string_columns=parquet_string_columns,
    )
    metadata_path = save_ingestion_metadata(metadata)
    evidence_paths.append(metadata_path)

    return {
        "dataframe": dataframe,
        "original_dataframe": preserved_copy,
        "column_mapping": original_by_normalized,
        "sheet_names": sheet_names,
        "dimensions": dimensions,
        "columns": columns,
        "data_types": data_types,
        "nulls": nulls,
        "nulls_per_row": nulls_per_row,
        "duplicate_summary": duplicate_table,
        "duplicate_examples": duplicate_examples,
        "identifiers": identifiers,
        "empty_constants": empty_constants,
        "text_problems": text_problems,
        "textual_nulls": textual_nulls,
        "category_summary": category_summary,
        "category_frequencies": category_frequencies,
        "category_equivalences": category_equivalences,
        "numeric_summary": numeric_summary,
        "numeric_outliers": numeric_outliers,
        "date_candidates": date_candidates,
        "date_results": date_results,
        "date_failures": date_failures,
        "temporal_summary": temporal_summary,
        "temporal_inconsistencies": temporal_inconsistencies,
        "classification": classification,
        "memory": memory,
        "validation_summary": validation_summary,
        "metadata": metadata,
        "evidence_paths": evidence_paths,
    }
