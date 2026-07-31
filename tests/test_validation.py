"""Prueba validaciones de calidad con DataFrames sintéticos."""

from pathlib import Path

import numpy as np
import pandas as pd

from src import validation


def synthetic_dataframe() -> pd.DataFrame:
    """Construye datos pequeños con condiciones de calidad conocidas."""
    return pd.DataFrame(
        {
            "id_paciente": ["001", "002", "002", None],
            "constante": ["A", "A", "A", "A"],
            "vacia": [None, None, None, None],
            "texto": [" A", "a ", "SIN DATO", "  "],
            "mixto": [1, "dos", 3, None],
            "categoria": ["Uno", "uno", "Úno", "Dos"],
            "numero": [0.0, 1.0, 2.0, 100.0],
            "fecha_ingreso": ["2024-01-01", "mal", "2024-01-03", "2024-01-04"],
            "fecha_salida": ["2024-01-02", "2024-01-02", "2024-01-01", "2024-01-04"],
        }
    )


def test_dimensions_and_nulls() -> None:
    """Dimensiones y nulos deben derivarse del DataFrame."""
    dataframe = synthetic_dataframe()
    dimensions = validation.calculate_dimensions(dataframe).iloc[0]
    nulls, distribution, summary = validation.analyze_nulls(dataframe)

    assert dimensions["filas"] == 4
    assert dimensions["columnas"] == 9
    assert dimensions["celdas_totales"] == 36
    assert dimensions["celdas_nulas"] == int(dataframe.isna().sum().sum())
    assert nulls.loc[nulls["columna"] == "vacia", "nulos"].iloc[0] == 4
    assert distribution["cantidad_filas"].sum() == 4
    assert summary["filas_con_al_menos_un_nulo"] > 0


def test_exact_duplicates() -> None:
    """Las filas duplicadas exactas deben contarse sin eliminarse."""
    dataframe = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    summary, examples, details = validation.analyze_exact_duplicates(dataframe)

    assert details["filas_duplicadas_exactas"] == 2
    assert details["grupos_duplicados"] == 1
    assert len(examples) == 2
    assert summary["filas_duplicadas_exactas"].iloc[0] == 2


def test_empty_constant_and_mixed_types() -> None:
    """Deben identificarse columnas vacías, constantes y de tipos mixtos."""
    dataframe = synthetic_dataframe()
    constants = validation.analyze_empty_constant_columns(dataframe)
    types = validation.analyze_data_types(dataframe)

    assert constants.loc[constants["columna"] == "vacia", "completamente_vacia"].iloc[0]
    assert constants.loc[constants["columna"] == "constante", "unico_valor_no_nulo"].iloc[0]
    assert types.loc[types["columna"] == "mixto", "contiene_tipos_mixtos"].iloc[0]


def test_text_spaces_and_textual_nulls() -> None:
    """Los espacios y marcadores textuales de nulos deben detectarse."""
    problems, textual_nulls = validation.analyze_text_problems(synthetic_dataframe())

    texto = problems.loc[problems["columna"] == "texto"].iloc[0]
    assert texto["espacios_iniciales"] > 0
    assert texto["espacios_finales"] > 0
    assert textual_nulls["cantidad"].sum() == 1


def test_categorical_statistics_and_equivalences() -> None:
    """Las frecuencias y categorías equivalentes deben calcularse."""
    summaries, frequencies, equivalences = validation.analyze_categorical_columns(
        synthetic_dataframe()
    )

    assert "categoria" in summaries["columna"].tolist()
    assert not frequencies.loc[frequencies["columna"] == "categoria"].empty
    assert "categoria" in equivalences["columna"].tolist()


def test_numeric_statistics_and_iqr() -> None:
    """Las estadísticas numéricas y extremos IQR deben ser reproducibles."""
    summaries, outliers = validation.analyze_numeric_columns(synthetic_dataframe())
    numeric = summaries.loc[summaries["columna"] == "numero"].iloc[0]

    assert numeric["cantidad"] == 4
    assert numeric["ceros"] == 1
    assert numeric["valores_extremos_iqr"] == 1
    assert outliers.loc[outliers["columna"] == "numero", "valor"].iloc[0] == 100


def test_date_detection_and_non_convertible_values() -> None:
    """Las candidatas a fecha deben conservar fallos de conversión como evidencia."""
    candidates, results, failures, converted = validation.detect_date_candidates(
        synthetic_dataframe()
    )

    assert {"fecha_ingreso", "fecha_salida"}.issubset(set(candidates["columna"]))
    ingreso = results.loc[results["columna"] == "fecha_ingreso"].iloc[0]
    assert ingreso["fechas_no_convertibles"] == 1
    assert failures.loc[failures["columna"] == "fecha_ingreso", "valor_original"].iloc[0] == "mal"
    assert "fecha_ingreso" in converted


def test_numeric_identifier_is_not_detected_as_date_by_content() -> None:
    """Una secuencia numérica sin nombre temporal no debe tratarse como fecha."""
    dataframe = pd.DataFrame({"identificacion": [1001, 1002, 1003]})

    candidates, _, _, _ = validation.detect_date_candidates(dataframe)

    assert candidates.empty


def test_temporal_consistency() -> None:
    """Una salida anterior al ingreso debe registrarse sin corregirse."""
    _, _, _, converted = validation.detect_date_candidates(synthetic_dataframe())
    summary, inconsistencies = validation.analyze_temporal_consistency(converted)

    pair = summary[
        (summary["columna_inicial"] == "fecha_ingreso")
        & (summary["columna_final"] == "fecha_salida")
    ].iloc[0]
    assert pair["fecha_final_anterior"] == 1
    assert len(inconsistencies) == 1


def test_csv_json_and_missing_plot_generation(tmp_path: Path) -> None:
    """Las evidencias CSV, JSON y PNG deben escribirse."""
    dataframe = synthetic_dataframe()
    csv_path = validation._write_csv(
        validation.calculate_dimensions(dataframe),
        tmp_path / "dimensions.csv",
    )
    json_path = tmp_path / "summary.json"
    from src.data_io import save_json

    save_json({"estado": "OK"}, json_path)
    image_path = validation.plot_missing_values(dataframe, tmp_path / "missing.png")

    assert csv_path.is_file()
    assert json_path.is_file()
    assert image_path.is_file()
    assert pd.read_csv(csv_path).iloc[0]["filas"] == 4
