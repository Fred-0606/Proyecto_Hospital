"""Construye los conjuntos analíticos de ingreso y salida sin fuga de información."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from src.categorical_cleaning import normalize_category_text
from src.paths import CONFIG_DIR, PROCESSED_DATA_DIR

with (CONFIG_DIR / "settings.yml").open(encoding="utf-8") as settings_file:
    _SETTINGS = yaml.safe_load(settings_file)

MODEL_A_THRESHOLD_HOURS = float(_SETTINGS["target"]["ingreso_hours"])
MODEL_B_THRESHOLD_HOURS = float(_SETTINGS["target"]["salida_hours"])
MAXIMUM_PLAUSIBLE_STAY_DAYS = 13
RECENT_ADMISSION_WINDOWS_HOURS = (3, 6, 24)

MODEL_A_OUTPUT_PATH = PROCESSED_DATA_DIR / "04_modelo_a.parquet"
MODEL_B_OUTPUT_PATH = PROCESSED_DATA_DIR / "04_modelo_b.parquet"

WEEKDAY_NAMES = {
    0: "lunes",
    1: "martes",
    2: "miercoles",
    3: "jueves",
    4: "viernes",
    5: "sabado",
    6: "domingo",
}
MONTH_NAMES = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}


def exclude_implausible_stays(
    dataframe: pd.DataFrame,
    maximum_days: int = MAXIMUM_PLAUSIBLE_STAY_DAYS,
) -> tuple[pd.DataFrame, int]:
    """Excluye episodios con alguna duración oficial superior al máximo clínico."""
    duration_columns = [
        "duracion_ingreso_calculada",
        "duracion_salida_calculada",
    ]
    exceeds_maximum = dataframe[duration_columns].gt(maximum_days * 24).any(axis=1)
    filtered = dataframe.loc[~exceeds_maximum].copy()
    return filtered, int(exceeds_maximum.sum())


def group_bed_category(value: object) -> str:
    """Reduce una cama individual a su área y tipo operativo."""
    normalized = normalize_category_text(value)
    if normalized in {"", "0", "INACTIVO", "CAMA SIN REGISTRAR"}:
        return "cama_sin_registrar"

    if normalized.startswith("HIDRA PX"):
        return "otras_camas"

    numbered_bed = re.fullmatch(r"(.+?) (CAMA|EXP|AISLAM|PX) \d+", normalized)
    if numbered_bed:
        area, bed_type = numbered_bed.groups()
        return f"{area}_{bed_type}".lower().replace(" ", "_")

    named_bed = re.fullmatch(r"B B([12])", normalized)
    if named_bed:
        return f"b_b{named_bed.group(1)}"

    return "otras_camas"


def add_calendar_features(
    dataframe: pd.DataFrame,
    date_column: str,
    suffix: str,
) -> pd.DataFrame:
    """Deriva hora, día, mes y fin de semana desde una fecha ya disponible."""
    result = dataframe.copy(deep=True)
    dates = pd.to_datetime(result[date_column], errors="coerce")

    result[f"hora_{suffix}"] = dates.dt.hour.astype("Int8")
    result[f"dia_semana_{suffix}"] = (
        dates.dt.dayofweek.map(WEEKDAY_NAMES).astype("string")
    )
    result[f"mes_{suffix}"] = dates.dt.month.map(MONTH_NAMES).astype("string")
    result[f"fin_semana_{suffix}"] = dates.dt.dayofweek.ge(5).astype("Int8")
    return result


def add_recent_admission_volume_features(
    dataframe: pd.DataFrame,
    date_column: str = "fecha_ingreso",
    windows_hours: tuple[int, ...] = RECENT_ADMISSION_WINDOWS_HOURS,
) -> pd.DataFrame:
    """Cuenta ingresos estrictamente anteriores dentro de ventanas recientes.

    Para cada registro se cuentan fechas en el intervalo [t - ventana, t). Al
    excluir el instante actual y cualquier fecha posterior, las variables están
    disponibles en el momento de predicción y no introducen fuga temporal.
    """
    result = dataframe.copy(deep=True)
    dates = pd.to_datetime(result[date_column], errors="coerce")
    sorted_dates = np.sort(dates.dropna().to_numpy(dtype="datetime64[ns]"))

    for hours in windows_hours:
        feature_name = f"volumen_ingresos_{hours}h"
        counts = pd.Series(pd.NA, index=result.index, dtype="Int32")
        valid_mask = dates.notna()
        if valid_mask.any():
            current_dates = dates.loc[valid_mask].to_numpy(dtype="datetime64[ns]")
            window_starts = current_dates - np.timedelta64(hours, "h")
            left_positions = np.searchsorted(sorted_dates, window_starts, side="left")
            right_positions = np.searchsorted(sorted_dates, current_dates, side="left")
            counts.loc[valid_mask] = right_positions - left_positions
        result[feature_name] = counts
    return result


def add_active_service_load_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la carga de otros pacientes antes de cada conducta.

    La carga en ingreso cuenta intervalos iniciados antes de ``t`` cuya conducta
    todavía no había finalizado, descontando el episodio evaluado. La carga de
    salida cuenta conductas estrictamente anteriores a ``t`` sin salida previa.
    Solo se usan intervalos con ambas marcas válidas y orden temporal coherente.
    """
    result = dataframe.copy(deep=True)
    reference_dates = pd.to_datetime(result["fecha_conducta"], errors="coerce")

    definitions = (
        (
            "pacientes_activos_en_ingreso",
            "fecha_ingreso",
            "fecha_conducta",
            True,
        ),
        (
            "pacientes_pendientes_de_salida",
            "fecha_conducta",
            "fecha_salida",
            False,
        ),
    )
    for feature_name, start_column, end_column, subtract_current in definitions:
        starts = pd.to_datetime(result[start_column], errors="coerce")
        ends = pd.to_datetime(result[end_column], errors="coerce")
        valid_intervals = starts.notna() & ends.notna() & ends.ge(starts)
        sorted_starts = np.sort(
            starts.loc[valid_intervals].to_numpy(dtype="datetime64[ns]")
        )
        sorted_ends = np.sort(
            ends.loc[valid_intervals].to_numpy(dtype="datetime64[ns]")
        )

        counts = pd.Series(pd.NA, index=result.index, dtype="Int32")
        valid_references = reference_dates.notna()
        if valid_references.any():
            references = reference_dates.loc[valid_references].to_numpy(
                dtype="datetime64[ns]"
            )
            active_counts = (
                np.searchsorted(sorted_starts, references, side="left")
                - np.searchsorted(sorted_ends, references, side="left")
            )
            if subtract_current:
                current_started_before = (
                    starts.loc[valid_references]
                    .lt(reference_dates.loc[valid_references])
                    .to_numpy(dtype="int8")
                )
                active_counts = active_counts - current_started_before
            counts.loc[valid_references] = np.maximum(active_counts, 0)
        result[feature_name] = counts
    return result


def build_model_a_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crea la tabla para predecir ingreso prolongado al finalizar el triage."""
    source_with_volume = add_recent_admission_volume_features(dataframe)
    valid_target = source_with_volume["duracion_ingreso_calculada"].notna()
    model_a = source_with_volume.loc[
        valid_target,
        [
            "fecha_ingreso",
            "triage",
            "edad",
            "afiliacion",
            "volumen_ingresos_3h",
            "volumen_ingresos_6h",
            "volumen_ingresos_24h",
            "duracion_ingreso_calculada",
        ],
    ].copy()
    model_a = add_calendar_features(model_a, "fecha_ingreso", "ingreso")
    model_a["estancia_prolongada_a"] = (
        model_a["duracion_ingreso_calculada"]
        .ge(MODEL_A_THRESHOLD_HOURS)
        .astype("int8")
    )
    model_a = model_a.rename(columns={"fecha_ingreso": "fecha_referencia"})
    return model_a[
        [
            "fecha_referencia",
            "triage",
            "edad",
            "afiliacion",
            "hora_ingreso",
            "dia_semana_ingreso",
            "mes_ingreso",
            "fin_semana_ingreso",
            "volumen_ingresos_3h",
            "volumen_ingresos_6h",
            "volumen_ingresos_24h",
            "duracion_ingreso_calculada",
            "estancia_prolongada_a",
        ]
    ].reset_index(drop=True)


def build_model_b_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crea la tabla para predecir salida prolongada al registrar la conducta."""
    source_with_load = add_active_service_load_features(dataframe)
    valid_rows = source_with_load["duracion_salida_calculada"].notna()
    model_b = source_with_load.loc[
        valid_rows,
        [
            "fecha_ingreso",
            "fecha_conducta",
            "triage",
            "edad",
            "afiliacion",
            "especialidades_tratantes",
            "sala_observacion",
            "cama",
            "conducta",
            "pacientes_activos_en_ingreso",
            "pacientes_pendientes_de_salida",
            "duracion_ingreso_calculada",
            "duracion_salida_calculada",
        ],
    ].copy()
    model_b = add_calendar_features(model_b, "fecha_ingreso", "ingreso")
    model_b = add_calendar_features(model_b, "fecha_conducta", "conducta")
    model_b["grupo_cama"] = model_b["cama"].map(group_bed_category).astype("string")
    model_b["estancia_prolongada_b"] = (
        model_b["duracion_salida_calculada"]
        .ge(MODEL_B_THRESHOLD_HOURS)
        .astype("int8")
    )
    model_b = model_b.rename(columns={"fecha_conducta": "fecha_referencia"})
    return model_b[
        [
            "fecha_referencia",
            "triage",
            "edad",
            "afiliacion",
            "especialidades_tratantes",
            "sala_observacion",
            "grupo_cama",
            "conducta",
            "pacientes_activos_en_ingreso",
            "pacientes_pendientes_de_salida",
            "duracion_ingreso_calculada",
            "hora_ingreso",
            "dia_semana_ingreso",
            "mes_ingreso",
            "fin_semana_ingreso",
            "hora_conducta",
            "dia_semana_conducta",
            "mes_conducta",
            "fin_semana_conducta",
            "duracion_salida_calculada",
            "estancia_prolongada_b",
        ]
    ].reset_index(drop=True)


def build_target_balance(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
) -> pd.DataFrame:
    """Resume cantidades y porcentajes de las clases de ambos objetivos."""
    frames: list[pd.DataFrame] = []
    for model_name, dataframe, target in (
        ("Modelo A — ingreso", model_a, "estancia_prolongada_a"),
        ("Modelo B — salida", model_b, "estancia_prolongada_b"),
    ):
        counts = dataframe[target].value_counts().reindex([0, 1], fill_value=0)
        summary = counts.rename_axis("clase").rename("registros").reset_index()
        summary["modelo"] = model_name
        summary["etiqueta"] = summary["clase"].map(
            {0: "Menos de 6 horas", 1: "6 horas o más"}
        )
        summary["porcentaje"] = summary["registros"].div(len(dataframe)).mul(100)
        frames.append(summary[["modelo", "clase", "etiqueta", "registros", "porcentaje"]])
    return pd.concat(frames, ignore_index=True)


def build_duration_summary(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula estadísticos robustos de los dos tiempos de interés."""
    rows: list[dict[str, float | int | str]] = []
    for model_name, series in (
        ("Tiempo A — ingreso", model_a["duracion_ingreso_calculada"]),
        ("Tiempo B — salida", model_b["duracion_salida_calculada"]),
    ):
        rows.append(
            {
                "tiempo": model_name,
                "registros": len(series),
                "media_horas": float(series.mean()),
                "mediana_horas": float(series.median()),
                "p25_horas": float(series.quantile(0.25)),
                "p75_horas": float(series.quantile(0.75)),
                "p95_horas": float(series.quantile(0.95)),
                "p99_horas": float(series.quantile(0.99)),
                "maximo_horas": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def build_duration_comparison(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compara duraciones originales y calculadas sin exponer episodios."""
    rows: list[dict[str, float | int | str | None]] = []
    definitions = (
        (
            "Tiempo A — ingreso",
            "tiempo_en_ingreso",
            "duracion_ingreso_calculada",
            MODEL_A_THRESHOLD_HOURS,
        ),
        (
            "Tiempo B — salida",
            "total_en_salida",
            "duracion_salida_calculada",
            MODEL_B_THRESHOLD_HOURS,
        ),
    )
    for name, original_column, calculated_column, threshold_hours in definitions:
        original = dataframe[original_column]
        calculated = dataframe[calculated_column]
        comparable = original.notna() & calculated.notna()
        absolute_difference = original.sub(calculated).abs()
        classification_changed = (
            original.ge(threshold_hours)
            .ne(calculated.ge(threshold_hours))
            & comparable
        )
        rows.append(
            {
                "tiempo": name,
                "columna_original": original_column,
                "columna_oficial": calculated_column,
                "casos_comparables": int(comparable.sum()),
                "diferencias_mayores_un_minuto": int(
                    (absolute_difference.gt(1 / 60) & comparable).sum()
                ),
                "diferencia_maxima_horas": (
                    float(absolute_difference[comparable].max())
                    if comparable.any()
                    else None
                ),
                "cambios_clasificacion_umbral_6h": int(
                    classification_changed.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_variable_dictionary() -> pd.DataFrame:
    """Documenta el papel de cada variable en las dos tablas analíticas."""
    rows = [
        ("fecha_referencia", "control", "A y B", "Ordenamiento y futura partición temporal; no entra al modelo."),
        ("triage", "predictora", "A y B", "Clasificación disponible al inicio del proceso."),
        ("edad", "predictora", "A y B", "Edad validada; su mediana se ajustará solo con entrenamiento."),
        ("afiliacion", "predictora", "A y B", "Afiliación registrada al inicio."),
        ("especialidades_tratantes", "predictora", "B", "Especialidades conocidas al registrar la conducta."),
        ("sala_observacion", "predictora", "B", "Ubicación conocida antes del proceso de salida."),
        ("grupo_cama", "predictora", "B", "Área y tipo de la primera cama disponible al momento B."),
        ("conducta", "predictora", "B", "Decisión registrada simultáneamente con el inicio del tiempo B."),
        ("duracion_ingreso_calculada", "objetivo/predictora", "A/B", "Duración oficial calculada desde fechas; objetivo continuo de A y antecedente observado en B."),
        ("hora_ingreso", "predictora derivada", "A y B", "Hora del ingreso posterior al triage."),
        ("dia_semana_ingreso", "predictora derivada", "A y B", "Día de la semana del ingreso."),
        ("mes_ingreso", "predictora derivada", "A y B", "Mes del ingreso para patrones estacionales."),
        ("fin_semana_ingreso", "predictora derivada", "A y B", "1 para sábado o domingo; 0 en otro caso."),
        ("volumen_ingresos_3h", "predictora derivada", "A", "Ingresos estrictamente anteriores ocurridos durante las 3 horas previas."),
        ("volumen_ingresos_6h", "predictora derivada", "A", "Ingresos estrictamente anteriores ocurridos durante las 6 horas previas."),
        ("volumen_ingresos_24h", "predictora derivada", "A", "Ingresos estrictamente anteriores ocurridos durante las 24 horas previas."),
        ("hora_conducta", "predictora derivada", "B", "Hora en que se registra la conducta."),
        ("dia_semana_conducta", "predictora derivada", "B", "Día de la semana de la conducta."),
        ("mes_conducta", "predictora derivada", "B", "Mes de la conducta."),
        ("fin_semana_conducta", "predictora derivada", "B", "1 para sábado o domingo; 0 en otro caso."),
        ("duracion_salida_calculada", "objetivo continuo", "B", "Duración oficial calculada entre conducta y salida."),
        ("estancia_prolongada_a", "objetivo binario", "A", "1 cuando la duración de ingreso es mayor o igual a 6 horas."),
        ("estancia_prolongada_b", "objetivo binario", "B", "1 cuando la duración de salida es mayor o igual a 6 horas."),
    ]
    return pd.DataFrame(rows, columns=["variable", "rol", "modelo", "justificacion"])


def build_categorical_cardinality(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
) -> pd.DataFrame:
    """Resume la cardinalidad de las predictoras categóricas finales."""
    rows: list[dict[str, int | str]] = []
    for model_name, dataframe, columns in (
        ("A", model_a, ["triage", "afiliacion", "dia_semana_ingreso", "mes_ingreso"]),
        (
            "B",
            model_b,
            [
                "triage",
                "afiliacion",
                "especialidades_tratantes",
                "sala_observacion",
                "grupo_cama",
                "conducta",
                "dia_semana_ingreso",
                "mes_ingreso",
                "dia_semana_conducta",
                "mes_conducta",
            ],
        ),
    ):
        for column in columns:
            rows.append(
                {
                    "modelo": model_name,
                    "variable": column,
                    "valores_unicos": int(dataframe[column].nunique(dropna=False)),
                }
            )
    return pd.DataFrame(rows)


def build_rate_by_category(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calcula la tasa de estancia prolongada por una categoría común."""
    frames: list[pd.DataFrame] = []
    for model_name, dataframe, target in (
        ("Modelo A — ingreso", model_a, "estancia_prolongada_a"),
        ("Modelo B — salida", model_b, "estancia_prolongada_b"),
    ):
        grouped = (
            dataframe.groupby(column, dropna=False, observed=True)[target]
            .agg(registros="size", tasa_prolongada="mean")
            .reset_index()
        )
        grouped["modelo"] = model_name
        grouped["tasa_prolongada"] *= 100
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def build_rate_by_reference_hour(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la tasa por hora disponible en cada momento de predicción."""
    frames: list[pd.DataFrame] = []
    for model_name, dataframe, hour, target in (
        ("Modelo A — ingreso", model_a, "hora_ingreso", "estancia_prolongada_a"),
        ("Modelo B — salida", model_b, "hora_conducta", "estancia_prolongada_b"),
    ):
        grouped = (
            dataframe.groupby(hour, observed=True)[target]
            .agg(registros="size", tasa_prolongada="mean")
            .reset_index(names="hora")
        )
        grouped["modelo"] = model_name
        grouped["tasa_prolongada"] *= 100
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def build_specialty_frequency(model_b: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Cuenta etiquetas individuales en la variable multiespecialidad del modelo B."""
    specialties = (
        model_b["especialidades_tratantes"]
        .str.split(" | ", regex=False)
        .explode()
        .dropna()
    )
    return (
        specialties.value_counts()
        .head(limit)
        .rename_axis("especialidad")
        .rename("registros")
        .reset_index()
    )


def build_duplicate_audit(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
) -> pd.DataFrame:
    """Resume duplicados exactos y coincidencias potenciales sin eliminarlos.

    Una coincidencia potencial comparte la fecha de referencia y todos los
    predictores, pero puede diferir en el objetivo continuo o binario. Sin un
    identificador persistente del episodio no se considera duplicado confirmado.
    """
    rows: list[dict[str, int | str]] = []
    definitions = (
        ("A", model_a, "estancia_prolongada_a", "duracion_ingreso_calculada"),
        ("B", model_b, "estancia_prolongada_b", "duracion_salida_calculada"),
    )
    for model_name, dataframe, target_column, outcome_duration in definitions:
        predictor_columns = dataframe.columns.difference(
            [target_column, outcome_duration],
            sort=False,
        ).tolist()
        for criterion, columns, interpretation, action in (
            (
                "fila_exacta",
                list(dataframe.columns),
                "Duplicado confirmado dentro de la tabla analitica.",
                "eliminar_registros_excedentes",
            ),
            (
                "fecha_y_predictores",
                predictor_columns,
                (
                    "Coincidencia potencial; requiere identificador del episodio "
                    "para confirmar duplicidad."
                ),
                "conservar_y_auditar",
            ),
        ):
            duplicated_mask = dataframe.duplicated(subset=columns, keep=False)
            duplicated_rows = dataframe.loc[duplicated_mask, columns]
            group_sizes = duplicated_rows.groupby(
                columns,
                dropna=False,
                observed=True,
            ).size()
            if criterion == "fecha_y_predictores":
                target_counts = dataframe.loc[duplicated_mask].groupby(
                    columns,
                    dropna=False,
                    observed=True,
                )[target_column].nunique(dropna=False)
                groups_with_discordant_target = int(target_counts.gt(1).sum())
            else:
                groups_with_discordant_target = 0
            rows.append(
                {
                    "modelo": model_name,
                    "criterio": criterion,
                    "columnas_comparadas": " | ".join(columns),
                    "grupos_duplicados": len(group_sizes),
                    "registros_en_grupos": int(group_sizes.sum()),
                    "registros_excedentes": int((group_sizes - 1).sum()),
                    "grupos_objetivo_discordante": groups_with_discordant_target,
                    "interpretacion": interpretation,
                    "accion_aplicada": action,
                }
            )
    return pd.DataFrame(rows)


def remove_exact_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Elimina solo copias completamente idénticas y reporta cuántas retira."""
    cleaned = dataframe.drop_duplicates().reset_index(drop=True)
    removed_rows = len(dataframe) - len(cleaned)
    return cleaned, removed_rows
