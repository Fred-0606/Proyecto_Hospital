"""Pruebas de las reglas activas de limpieza y calidad."""

import pandas as pd

from src.categorical_cleaning import (
    canonicalize_affiliation,
    canonicalize_disposition,
    canonicalize_room,
    canonicalize_specialties,
    canonicalize_triage,
    clean_selected_categories,
)
from src.cleaning import (
    convert_selected_column_types,
    duration_to_hours,
    encode_columns_for_variance_threshold,
    impute_missing_bed_category,
    recalculate_stay_durations,
    to_nullable_integer,
)


def test_variance_encoding_preserves_columns_and_marks_nulls() -> None:
    """La codificación temporal debe conservar columnas y representar nulos."""
    dataframe = pd.DataFrame(
        {
            "constante": ["A", "A", None],
            "variable": ["A", "B", None],
        }
    )

    encoded = encode_columns_for_variance_threshold(dataframe)

    assert encoded.columns.tolist() == dataframe.columns.tolist()
    assert encoded.iloc[:2].notna().all().all()
    assert encoded.iloc[2].isna().all()


def test_age_conversion_accepts_only_integer_numbers() -> None:
    """Edad debe ser un entero entre 0 y 120; lo demás debe ser nulo."""
    result = to_nullable_integer(
        pd.Series(["20", 35, "18.5", "sin dato", None, -1, 121, 0, 120])
    )

    assert str(result.dtype) == "Int64"
    assert result.tolist()[:2] == [20, 35]
    assert result.isna().tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_duration_conversion_returns_decimal_hours() -> None:
    """Las duraciones hh:mm y hh:mm:ss deben convertirse en horas decimales."""
    result = duration_to_hours(pd.Series(["06:30", "1:15:00", "inválido", None]))

    assert result.iloc[0] == 6.5
    assert result.iloc[1] == 1.25
    assert result.iloc[2:].isna().all()


def test_selected_type_conversion_reports_new_nulls() -> None:
    """La auditoría debe registrar valores incompatibles convertidos a nulo."""
    dataframe = pd.DataFrame(
        {
            "edad": ["20", "20.5"],
            "tiempo_en_ingreso": ["06:30", "mal"],
            "total_en_salida": ["01:00", "02:30"],
            "fecha_ingreso": ["2024-01-01 10:00", "mal"],
            "fecha_conducta": ["2024-01-01 16:30", "2024-01-01 12:00"],
            "fecha_salida": ["2024-01-01 17:30", "2024-01-01 14:30"],
        }
    )

    converted, audit = convert_selected_column_types(dataframe)
    audit = audit.set_index("columna")

    assert str(converted["edad"].dtype) == "Int64"
    assert converted.loc[0, "tiempo_en_ingreso"] == 6.5
    assert pd.api.types.is_datetime64_any_dtype(converted["fecha_ingreso"])
    assert audit.loc["edad", "nulos_nuevos"] == 1
    assert audit.loc["tiempo_en_ingreso", "nulos_nuevos"] == 1


def test_missing_bed_uses_explicit_category_without_mutating_source() -> None:
    """Los nulos de cama deben convertirse en una categoría trazable."""
    source = pd.DataFrame({"cama": ["A-01", None, pd.NA]})

    result = impute_missing_bed_category(source)

    assert result["cama"].tolist() == [
        "A-01",
        "cama_sin_registrar",
        "cama_sin_registrar",
    ]
    assert source["cama"].isna().sum() == 2


def test_stay_durations_are_recalculated_and_negatives_become_null() -> None:
    """Los intervalos finales deben provenir de fechas y rechazar negativos."""
    dataframe = pd.DataFrame(
        {
            "fecha_ingreso": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 12:00"]),
            "fecha_conducta": pd.to_datetime(["2024-01-01 14:30", "2024-01-02 10:00"]),
            "fecha_salida": pd.to_datetime(["2024-01-01 16:00", "2024-01-02 15:00"]),
            "tiempo_en_ingreso": [99.0, 2.0],
            "total_en_salida": [None, 5.0],
        }
    )

    result, audit = recalculate_stay_durations(dataframe)
    audit = audit.set_index("duracion_calculada")

    assert result.loc[0, "duracion_ingreso_calculada"] == 6.5
    assert pd.isna(result.loc[1, "duracion_ingreso_calculada"])
    assert result.loc[0, "duracion_salida_calculada"] == 1.5
    assert result.loc[0, "tiempo_en_ingreso"] == 99.0
    assert pd.isna(result.loc[0, "total_en_salida"])
    assert audit.loc["duracion_ingreso_calculada", "duraciones_negativas"] == 1
    assert audit.loc[
        "duracion_salida_calculada", "recuperados_desde_fechas"
    ] == 1


def test_specialties_are_homologated_as_ordered_multilabels() -> None:
    """Variantes y combinaciones deben usar un vocabulario canónico."""
    assert canonicalize_specialties("MEFA / PSIQUIATRÍA") == (
        "medicina_familiar | psiquiatria"
    )
    assert canonicalize_specialties("servicio desconocido") == "otra_especialidad"
    assert canonicalize_specialties(None) == "especialidad_sin_registrar"


def test_room_triage_affiliation_and_disposition_rules() -> None:
    """Las reglas acordadas deben producir categorías explícitas."""
    assert canonicalize_room("OBS- CAMA 4") == "sala_sin_registrar"
    assert canonicalize_room(None) == "sala_sin_registrar"
    assert canonicalize_triage("Triage 5") == "triage_5"
    assert canonicalize_triage("Seleccionar") == "triage_sin_registrar"
    assert canonicalize_affiliation("INACTIVO") == "inactivo"
    assert canonicalize_affiliation("Seleccionar") == "afiliacion_sin_registrar"
    assert canonicalize_disposition("REMISION") == "hospitalizacion"
    assert canonicalize_disposition("SA") == "salida"
    assert canonicalize_disposition(None) == "conducta_sin_registrar"


def test_category_cleaning_preserves_rows_and_returns_audits() -> None:
    """La homologación debe conservar filas y documentar el vocabulario."""
    dataframe = pd.DataFrame(
        {
            "especialidades_tratantes": ["MED INTERNA", None],
            "sala_observacion": ["NORTE - Sala A", "OBS- CAMA 4"],
            "triage": ["Triage 2", "Seleccionar"],
            "afiliacion": ["INACTIVO", None],
            "conducta": ["ORDEN SALIDA", "REMISION"],
        }
    )

    result, audit, vocabulary = clean_selected_categories(dataframe)

    assert len(result) == len(dataframe)
    assert len(audit) == 5
    assert audit.loc[
        audit["columna"].eq("especialidades_tratantes"),
        "etiquetas_individuales_finales",
    ].iloc[0] == 2
    assert set(vocabulary["especialidad_homologada"]) == {
        "especialidad_sin_registrar",
        "medicina_interna",
    }
