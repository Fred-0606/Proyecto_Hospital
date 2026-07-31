"""Gestiona la ingesta inmutable de la hoja CENSO y sus artefactos técnicos."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.paths import INTERIM_DATA_DIR, METRICS_DIR, PROJECT_ROOT, RAW_DATA_DIR


SOURCE_FILENAME = "database_modificado.xlsx"
REQUIRED_SHEET = "CENSO"


def get_source_excel_path() -> Path:
    """Devuelve la ruta portable del archivo Excel fuente."""
    return RAW_DATA_DIR / SOURCE_FILENAME


def require_file(path: Path, instruction: str) -> Path:
    """Valida que un archivo exista y produce un error operativo claro."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}. {instruction}")
    return path


def get_excel_sheet_names(excel_path: Path) -> list[str]:
    """Obtiene nombres de hojas sin cargar sus contenidos como DataFrames."""
    require_file(excel_path, "Ubique el archivo fuente antes de ejecutar la etapa 01.")
    with pd.ExcelFile(excel_path) as workbook:
        return list(workbook.sheet_names)


def require_sheet(
    excel_path: Path,
    sheet_names: list[str],
    required_sheet: str = REQUIRED_SHEET,
) -> None:
    """Comprueba la presencia exacta de una hoja y detalla el libro revisado."""
    if required_sheet not in sheet_names:
        raise ValueError(
            "No se encontró la hoja requerida. "
            f"Archivo revisado: {excel_path}; "
            f"hojas encontradas: {sheet_names}; "
            f"hoja requerida: {required_sheet}."
        )


def read_censo_sheet(excel_path: Path | None = None) -> pd.DataFrame:
    """Lee solo CENSO, descarta la primera fila y usa la segunda como encabezado."""
    source_path = excel_path or get_source_excel_path()
    sheet_names = get_excel_sheet_names(source_path)
    require_sheet(source_path, sheet_names)
    return pd.read_excel(source_path, sheet_name=REQUIRED_SHEET, header=1)


def copy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crea una copia profunda e independiente de un DataFrame."""
    return dataframe.copy(deep=True)


def normalize_column_name(column_name: object) -> str:
    """Normaliza un nombre de columna a minúsculas ASCII con guiones bajos."""
    text = str(column_name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "columna"


def normalize_dataframe_columns(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normaliza columnas y resuelve colisiones con sufijos deterministas."""
    normalized_dataframe = copy_dataframe(dataframe)
    occurrences: dict[str, int] = {}
    normalized_names: list[str] = []
    original_by_normalized: dict[str, str] = {}

    for original_name in dataframe.columns:
        base_name = normalize_column_name(original_name)
        occurrences[base_name] = occurrences.get(base_name, 0) + 1
        occurrence = occurrences[base_name]
        normalized_name = base_name if occurrence == 1 else f"{base_name}__{occurrence}"
        normalized_names.append(normalized_name)
        original_by_normalized[normalized_name] = str(original_name)

    normalized_dataframe.columns = normalized_names
    return normalized_dataframe, original_by_normalized


def identify_mixed_object_columns(dataframe: pd.DataFrame) -> list[str]:
    """Identifica columnas object con más de un tipo Python no nulo."""
    return [
        column
        for column in dataframe.columns
        if dataframe[column].dtype == object
        if dataframe[column].dropna().map(type).nunique() > 1
    ]


def prepare_parquet_copy(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Crea una copia compatible con Parquet sin alterar el DataFrame analizado."""
    parquet_dataframe = copy_dataframe(dataframe)
    mixed_columns = identify_mixed_object_columns(dataframe)
    for column in mixed_columns:
        parquet_dataframe[column] = dataframe[column].map(
            lambda value: str(value) if pd.notna(value) else None
        ).astype("string")
    return parquet_dataframe, mixed_columns


def save_dataframe_as_parquet(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Guarda una copia compatible con Parquet sin mutar el DataFrame recibido."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_dataframe, _ = prepare_parquet_copy(dataframe)
    parquet_dataframe.to_parquet(output_path, index=False)
    return output_path


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula la huella SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative_path(path: Path) -> str:
    """Representa una ruta relativa a la raíz del proyecto con separadores POSIX."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _json_default(value: Any) -> Any:
    """Convierte tipos frecuentes de pandas, NumPy y fechas a JSON."""
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return project_relative_path(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"El tipo {type(value).__name__} no es serializable en JSON.")


def save_json(payload: dict[str, Any] | list[Any], output_path: Path) -> Path:
    """Guarda metadatos o métricas en JSON UTF-8 de forma legible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(
            payload,
            destination,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    return output_path


def build_ingestion_metadata(
    *,
    source_path: Path,
    sheet_names: list[str],
    dataframe: pd.DataFrame,
    parquet_path: Path,
    evidence_paths: list[Path],
    parquet_string_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Construye metadatos verificables de la ingesta ejecutada."""
    rows, columns = dataframe.shape
    return {
        "fecha_hora_ejecucion": datetime.now().astimezone().isoformat(),
        "version_python": __import__("platform").python_version(),
        "version_pandas": pd.__version__,
        "ruta_relativa_archivo": project_relative_path(source_path),
        "nombre_archivo": source_path.name,
        "sha256": calculate_sha256(source_path),
        "tamano_archivo_bytes": source_path.stat().st_size,
        "hoja_utilizada": REQUIRED_SHEET,
        "hojas_ignoradas": [name for name in sheet_names if name != REQUIRED_SHEET],
        "dimensiones": {"filas": rows, "columnas": columns},
        "numero_columnas": columns,
        "numero_registros": rows,
        "numero_total_nulos": int(dataframe.isna().sum().sum()),
        "numero_duplicados_exactos": int(dataframe.duplicated(keep=False).sum()),
        "ruta_parquet": project_relative_path(parquet_path),
        "columnas_serializadas_como_texto_en_parquet": parquet_string_columns or [],
        "archivos_producidos": sorted(
            project_relative_path(path) for path in evidence_paths
        ),
    }


def save_ingestion_metadata(metadata: dict[str, Any]) -> Path:
    """Guarda los metadatos de ingesta en su ruta estándar."""
    return save_json(metadata, METRICS_DIR / "01_metadatos_ingesta.json")


def get_identified_parquet_path() -> Path:
    """Devuelve la ruta estándar de la salida principal de la etapa 01."""
    return INTERIM_DATA_DIR / "01_censo_identificado.parquet"
