"""Pruebas de los conjuntos analíticos y variables temporales de la etapa 04."""

import pandas as pd

from src.features import (
    add_calendar_features,
    add_active_service_load_features,
    add_recent_admission_volume_features,
    build_duplicate_audit,
    build_model_a_dataset,
    build_model_b_dataset,
    exclude_implausible_stays,
    group_bed_category,
    remove_exact_duplicates,
)


def _sample_dataframe() -> pd.DataFrame:
    """Crea casos mínimos válidos e inválidos para ambos modelos."""
    return pd.DataFrame(
        {
            "triage": pd.Series(["triage_2", "triage_3", "triage_4"], dtype="string"),
            "edad": pd.Series([30, pd.NA, 70], dtype="Int64"),
            "afiliacion": pd.Series(["a", "b", "c"], dtype="string"),
            "especialidades_tratantes": pd.Series(
                ["medicina_interna", "cirugia_general", "pediatria"], dtype="string"
            ),
            "sala_observacion": pd.Series(["sala_1", "sala_2", "sala_3"], dtype="string"),
            "cama": pd.Series(["cama_1", "cama_2", "cama_3"], dtype="string"),
            "fecha_ingreso": pd.to_datetime(
                ["2026-01-03 08:00", "2026-01-05 09:00", "2026-01-06 10:00"]
            ),
            "duracion_ingreso_calculada": [6.0, 7.0, float("nan")],
            "fecha_conducta": pd.to_datetime(
                ["2026-01-03 14:00", "2026-01-05 16:00", "2026-01-06 11:00"]
            ),
            "conducta": pd.Series(["salida", "hospitalizacion", "salida"], dtype="string"),
            "duracion_salida_calculada": [6.0, 8.0, 9.0],
            "fecha_salida": pd.to_datetime(
                ["2026-01-03 20:00", "2026-01-06 00:00", "2026-01-06 20:00"]
            ),
        }
    )


def test_calendar_features_identify_weekend() -> None:
    """Sábado y domingo deben codificarse como fin de semana."""
    dataframe = pd.DataFrame(
        {"fecha": pd.to_datetime(["2026-01-03 08:00", "2026-01-05 09:00"])}
    )

    result = add_calendar_features(dataframe, "fecha", "evento")

    assert result["hora_evento"].tolist() == [8, 9]
    assert result["dia_semana_evento"].tolist() == ["sabado", "lunes"]
    assert result["mes_evento"].tolist() == ["enero", "enero"]
    assert result["fin_semana_evento"].tolist() == [1, 0]


def test_recent_admission_volumes_use_only_strictly_previous_events() -> None:
    """Las ventanas deben excluir el registro actual, empates y eventos futuros."""
    dataframe = pd.DataFrame(
        {
            "fecha_ingreso": pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-01 08:00",
                    "2026-01-01 10:00",
                    "2025-12-31 10:00",
                ]
            )
        }
    )

    result = add_recent_admission_volume_features(dataframe)

    assert result["volumen_ingresos_3h"].tolist() == [1, 0, 1, 0]
    assert result["volumen_ingresos_6h"].tolist() == [1, 0, 1, 0]
    assert result["volumen_ingresos_24h"].tolist() == [2, 1, 2, 0]


def test_active_service_load_excludes_current_and_future_patients() -> None:
    """La carga B debe representar otros pacientes conocidos antes de la conducta."""
    dataframe = pd.DataFrame(
        {
            "fecha_ingreso": pd.to_datetime(
                ["2026-01-01 08:00", "2026-01-01 09:00", "2026-01-01 11:00"]
            ),
            "fecha_conducta": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 12:00", "2026-01-01 13:00"]
            ),
            "fecha_salida": pd.to_datetime(
                ["2026-01-01 15:00", "2026-01-01 14:00", "2026-01-01 16:00"]
            ),
        }
    )

    result = add_active_service_load_features(dataframe)

    assert result["pacientes_activos_en_ingreso"].tolist() == [1, 1, 0]
    assert result["pacientes_pendientes_de_salida"].tolist() == [0, 1, 2]


def test_model_a_includes_six_hour_threshold_and_excludes_null_target() -> None:
    """Seis horas es prolongada y un objetivo nulo no genera etiqueta."""
    result = build_model_a_dataset(_sample_dataframe())

    assert len(result) == 2
    assert result["estancia_prolongada_a"].tolist() == [1, 1]
    assert "fecha_conducta" not in result.columns
    assert "fecha_salida" not in result.columns
    assert result["volumen_ingresos_3h"].tolist() == [0, 0]
    assert result["volumen_ingresos_6h"].tolist() == [0, 0]
    assert result["volumen_ingresos_24h"].tolist() == [0, 0]


def test_model_b_requires_only_valid_target() -> None:
    """El modelo B conserva faltantes predictivos para tratarlos tras la partición."""
    result = build_model_b_dataset(_sample_dataframe())

    assert len(result) == 3
    assert result["estancia_prolongada_b"].tolist() == [1, 1, 1]
    assert pd.isna(result.loc[2, "duracion_ingreso_calculada"])
    assert "fecha_salida" not in result.columns
    assert "fecha_conducta" not in result.columns
    assert "cama" not in result.columns
    assert "especialidades_tratantes" not in result.columns
    assert result["grupo_cama"].tolist() == ["otras_camas"] * 3
    assert result["fin_semana_conducta"].tolist() == [1, 0, 0]
    assert result["pacientes_activos_en_ingreso"].tolist() == [0, 0, 0]
    assert result["pacientes_pendientes_de_salida"].tolist() == [0, 0, 0]


def test_duplicate_audit_distinguishes_exact_and_potential_matches() -> None:
    """La auditoria no debe confundir predictores iguales con filas identicas."""
    model_a = build_model_a_dataset(_sample_dataframe())
    exact_copy = model_a.iloc[[0]].copy()
    different_outcome = model_a.iloc[[0]].copy()
    different_outcome["duracion_ingreso_calculada"] = 5.0
    different_outcome["estancia_prolongada_a"] = 0
    model_a = pd.concat(
        [model_a, exact_copy, different_outcome],
        ignore_index=True,
    )
    model_b = build_model_b_dataset(_sample_dataframe())

    audit = build_duplicate_audit(model_a, model_b).set_index(
        ["modelo", "criterio"]
    )

    exact = audit.loc[("A", "fila_exacta")]
    potential = audit.loc[("A", "fecha_y_predictores")]
    assert exact["grupos_duplicados"] == 1
    assert exact["registros_en_grupos"] == 2
    assert exact["registros_excedentes"] == 1
    assert potential["grupos_duplicados"] == 1
    assert potential["registros_en_grupos"] == 3
    assert potential["registros_excedentes"] == 2
    assert potential["grupos_objetivo_discordante"] == 1
    assert audit.loc[("B", "fila_exacta"), "grupos_duplicados"] == 0


def test_remove_exact_duplicates_keeps_one_copy() -> None:
    """La depuración retira solo copias idénticas y conserva una observación."""
    source = pd.DataFrame({"valor": [1, 1, 2], "clase": [0, 0, 1]})

    cleaned, removed_rows = remove_exact_duplicates(source)

    assert cleaned.to_dict(orient="records") == [
        {"valor": 1, "clase": 0},
        {"valor": 2, "clase": 1},
    ]
    assert removed_rows == 1


def test_implausible_stays_above_thirteen_days_are_excluded() -> None:
    """Trece días se conservan y cualquier duración superior se excluye."""
    dataframe = pd.DataFrame(
        {
            "duracion_ingreso_calculada": [312.0, 312.1, 20.0],
            "duracion_salida_calculada": [10.0, 20.0, 313.0],
        }
    )

    filtered, removed_rows = exclude_implausible_stays(dataframe)

    assert filtered.index.tolist() == [0]
    assert removed_rows == 2


def test_bed_category_is_grouped_by_area_and_type() -> None:
    """El número individual no debe crear categorías diferentes de cama."""
    assert group_bed_category("A- CAMA 01") == "a_cama"
    assert group_bed_category("A- EXP 14") == "a_exp"
    assert group_bed_category("B- AISLAM 3") == "b_aislam"
    assert group_bed_category("B- B1") == "b_b1"
    assert group_bed_category("0") == "cama_sin_registrar"
    assert group_bed_category("INACTIVO") == "cama_sin_registrar"
    assert group_bed_category("HIDRA - Px 1") == "otras_camas"
    assert group_bed_category("formato extraño") == "otras_camas"
