"""Entrena, serializa y sirve inferencias de los modelos clínicos congelados."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_io import require_file
from src.categorical_cleaning import clean_selected_categories
from src.cleaning import (
    convert_selected_column_types,
    impute_missing_bed_category,
    recalculate_stay_durations,
)
from src.data_io import harmonize_source_columns, normalize_dataframe_columns
from src.features import (
    add_calendar_features,
    group_bed_category,
)
from src.model_freeze import build_frozen_pipeline, load_frozen_configuration
from src.paths import MODELS_DIR
from src.preprocessing import (
    MODEL_A_DEVELOPMENT_PATH,
    MODEL_A_PREDICTORS,
    MODEL_A_TARGET,
    MODEL_B_DEVELOPMENT_PATH,
    MODEL_B_PREDICTORS,
    MODEL_B_TARGET,
)

MODEL_PATHS = {
    "A": MODELS_DIR / "modelo_a.joblib",
    "B": MODELS_DIR / "modelo_b.joblib",
}
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"
CENSO_COLUMNS = [
    "triage",
    "edad",
    "afiliacion",
    "especialidades_tratantes",
    "sala_observacion",
    "cama",
    "fecha_ingreso",
    "tiempo_en_ingreso",
    "fecha_conducta",
    "conducta",
    "total_en_salida",
    "fecha_salida",
]


def _sha256(path: Path) -> str:
    """Calcula la huella SHA-256 de un archivo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_and_serialize_models() -> dict[str, Any]:
    """Ajusta en desarrollo y persiste los dos pipelines congelados completos."""
    configuration = load_frozen_configuration()
    contracts = {
        "A": (MODEL_A_DEVELOPMENT_PATH, MODEL_A_PREDICTORS, MODEL_A_TARGET),
        "B": (MODEL_B_DEVELOPMENT_PATH, MODEL_B_PREDICTORS, MODEL_B_TARGET),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"random_state": 42, "models": {}}

    for model_name, (dataset_path, predictors, target) in contracts.items():
        require_file(
            dataset_path,
            "Ejecute primero la etapa 04 para crear la partición de desarrollo.",
        )
        development = pd.read_parquet(dataset_path)
        missing = sorted(set([*predictors, target]).difference(development.columns))
        if missing:
            raise ValueError(
                f"Faltan columnas para entrenar el Modelo {model_name}: {missing}."
            )
        pipeline = build_frozen_pipeline(model_name, configuration)
        pipeline.fit(development[predictors], development[target])
        output_path = MODEL_PATHS[model_name]
        joblib.dump(pipeline, output_path, compress=3)
        metadata["models"][model_name] = {
            "path": output_path.name,
            "sha256": _sha256(output_path),
            "training_rows": int(len(development)),
            "predictors": predictors,
            "target": target,
            "threshold": float(configuration["modelos"][model_name]["umbral"]),
        }

    MODEL_METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_serialized_models() -> tuple[dict[str, Pipeline], dict[str, Any]]:
    """Carga modelos verificados y su contrato persistido."""
    require_file(
        MODEL_METADATA_PATH,
        "Ejecute scripts/train_inference_models.py antes de iniciar la aplicación.",
    )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    models: dict[str, Pipeline] = {}
    for model_name, path in MODEL_PATHS.items():
        require_file(path, "Entrene y serialice primero los modelos de inferencia.")
        expected_hash = metadata["models"][model_name]["sha256"]
        if _sha256(path) != expected_hash:
            raise ValueError(f"La huella del Modelo {model_name} no coincide.")
        models[model_name] = joblib.load(path)
    return models, metadata


def predict_patient(
    model_name: str,
    patient: dict[str, Any],
    models: dict[str, Pipeline],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Devuelve probabilidad y clasificación binaria para un paciente."""
    if model_name not in {"A", "B"}:
        raise ValueError("model debe ser 'A' o 'B'.")
    contract = metadata["models"][model_name]
    predictors = contract["predictors"]
    missing = sorted(set(predictors).difference(patient))
    if missing:
        raise ValueError(f"Faltan variables para el Modelo {model_name}: {missing}.")
    row = pd.DataFrame([{name: patient[name] for name in predictors}])
    probability = float(models[model_name].predict_proba(row)[0, 1])
    threshold = float(contract["threshold"])
    positive = probability >= threshold
    return {
        "model": model_name,
        "target": contract["target"],
        "probability": probability,
        "threshold": threshold,
        "prediction": int(positive),
        "label": "SÍ" if positive else "NO",
        "definition": "estancia de 6 horas o más",
    }


def prepare_censo_context(content: bytes) -> pd.DataFrame:
    """Valida y transforma CENSO en memoria sin persistir datos hospitalarios."""
    if not content:
        raise ValueError("El archivo Excel está vacío.")
    workbook = pd.ExcelFile(io.BytesIO(content))
    if "CENSO" not in workbook.sheet_names:
        raise ValueError(
            "No se encontró la hoja requerida CENSO. "
            f"Hojas encontradas: {workbook.sheet_names}."
        )
    source = pd.read_excel(workbook, sheet_name="CENSO", header=1)
    harmonized = harmonize_source_columns(source)
    normalized, _ = normalize_dataframe_columns(harmonized)
    missing = sorted(set(CENSO_COLUMNS).difference(normalized.columns))
    if missing:
        raise ValueError(f"La hoja CENSO no cumple el contrato: faltan {missing}.")
    selected = normalized[CENSO_COLUMNS].copy(deep=True)
    converted, _ = convert_selected_column_types(selected)
    converted, _ = recalculate_stay_durations(converted)
    converted = impute_missing_bed_category(converted)
    cleaned, _, _ = clean_selected_categories(converted)
    return cleaned.reset_index(drop=True)


def summarize_censo_context(context: pd.DataFrame) -> dict[str, Any]:
    """Devuelve únicamente información agregada y rangos seguros del contexto."""
    eligible_a = context.index[context["fecha_ingreso"].notna()].tolist()
    eligible_b = context.index[context["fecha_conducta"].notna()].tolist()
    return {
        "rows": int(len(context)),
        "columns": int(len(context.columns)),
        "eligible": {"A": len(eligible_a), "B": len(eligible_b)},
        "first_episode": {
            "A": int(eligible_a[0]) if eligible_a else None,
            "B": int(eligible_b[0]) if eligible_b else None,
        },
        "options": {
            column: sorted(context[column].dropna().astype(str).unique().tolist())
            for column in (
                "triage",
                "afiliacion",
                "sala_observacion",
                "cama",
                "conducta",
            )
        },
    }


def build_episode_predictors(
    context: pd.DataFrame,
    model_name: str,
    episode_index: int,
) -> dict[str, Any]:
    """Construye predictores observados y derivados para un episodio de CENSO."""
    if model_name not in {"A", "B"}:
        raise ValueError("model debe ser 'A' o 'B'.")
    if episode_index not in context.index:
        raise ValueError("El episodio solicitado no existe en el contexto CENSO.")
    row = context.loc[episode_index]
    admission = pd.to_datetime(row["fecha_ingreso"], errors="coerce")
    if pd.isna(admission):
        raise ValueError("El episodio no tiene una fecha de ingreso válida.")
    base: dict[str, Any] = {
        "edad": row["edad"],
        "triage": row["triage"],
        "afiliacion": row["afiliacion"],
    }
    admission_calendar = add_calendar_features(
        pd.DataFrame({"fecha_ingreso": [admission]}),
        "fecha_ingreso",
        "ingreso",
    ).iloc[0]
    for name in (
        "hora_ingreso",
        "dia_semana_ingreso",
        "mes_ingreso",
        "fin_semana_ingreso",
    ):
        base[name] = admission_calendar[name]

    if model_name == "A":
        ingress_dates = pd.to_datetime(context["fecha_ingreso"], errors="coerce")
        for hours in (3, 6, 24):
            start = admission - pd.Timedelta(hours=hours)
            base[f"volumen_ingresos_{hours}h"] = int(
                ingress_dates.ge(start).mul(ingress_dates.lt(admission)).sum()
            )
        return base

    conduct_date = pd.to_datetime(row["fecha_conducta"], errors="coerce")
    if pd.isna(conduct_date):
        raise ValueError("El episodio no tiene una fecha de conducta válida.")
    conduct_calendar = add_calendar_features(
        pd.DataFrame({"fecha_conducta": [conduct_date]}),
        "fecha_conducta",
        "conducta",
    ).iloc[0]
    for name in (
        "hora_conducta",
        "dia_semana_conducta",
        "mes_conducta",
        "fin_semana_conducta",
    ):
        base[name] = conduct_calendar[name]

    ingress = pd.to_datetime(context["fecha_ingreso"], errors="coerce")
    conducts = pd.to_datetime(context["fecha_conducta"], errors="coerce")
    exits = pd.to_datetime(context["fecha_salida"], errors="coerce")
    valid_ingress_intervals = ingress.notna() & conducts.notna() & conducts.ge(ingress)
    valid_exit_intervals = conducts.notna() & exits.notna() & exits.ge(conducts)
    active = valid_ingress_intervals & ingress.lt(conduct_date) & conducts.ge(conduct_date)
    pending = valid_exit_intervals & conducts.lt(conduct_date) & exits.ge(conduct_date)
    active.loc[episode_index] = False
    pending.loc[episode_index] = False
    base.update(
        {
            "pacientes_activos_en_ingreso": int(active.sum()),
            "pacientes_pendientes_de_salida": int(pending.sum()),
            "duracion_ingreso_calculada": float(
                (conduct_date - admission).total_seconds() / 3600
            ),
            "sala_observacion": row["sala_observacion"],
            "grupo_cama": group_bed_category(row["cama"]),
            "conducta": row["conducta"],
        }
    )
    return base


def predict_episode(
    model_name: str,
    episode_index: int,
    context: pd.DataFrame,
    models: dict[str, Pipeline],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Predice un episodio real y añade trazabilidad de variables automáticas."""
    predictors = build_episode_predictors(context, model_name, episode_index)
    result = predict_patient(model_name, predictors, models, metadata)
    result["episode_index"] = episode_index
    automatic_features: dict[str, Any] = {}
    for name, value in predictors.items():
        if pd.isna(value):
            automatic_features[name] = None
        else:
            automatic_features[name] = value.item() if hasattr(value, "item") else value
    result["automatic_features"] = automatic_features
    return result


def build_new_patient_predictors(
    context: pd.DataFrame,
    model_name: str,
    patient: dict[str, Any],
) -> dict[str, Any]:
    """Combina datos observados del paciente nuevo con contexto derivado de CENSO."""
    common_required = {"edad", "triage", "afiliacion", "fecha_ingreso"}
    model_b_required = {"fecha_conducta", "conducta", "sala_observacion", "cama"}
    required = common_required | (model_b_required if model_name == "B" else set())
    missing = sorted(name for name in required if patient.get(name) in (None, ""))
    if missing:
        raise ValueError(f"Faltan datos observados del paciente: {missing}.")

    admission = pd.to_datetime(patient["fecha_ingreso"], errors="coerce")
    if pd.isna(admission):
        raise ValueError("fecha_ingreso no es una fecha válida.")
    base: dict[str, Any] = {
        "edad": patient["edad"],
        "triage": patient["triage"],
        "afiliacion": patient["afiliacion"],
    }
    admission_calendar = add_calendar_features(
        pd.DataFrame({"fecha_ingreso": [admission]}),
        "fecha_ingreso",
        "ingreso",
    ).iloc[0]
    for name in (
        "hora_ingreso",
        "dia_semana_ingreso",
        "mes_ingreso",
        "fin_semana_ingreso",
    ):
        base[name] = admission_calendar[name]

    ingress_dates = pd.to_datetime(context["fecha_ingreso"], errors="coerce")
    if model_name == "A":
        for hours in (3, 6, 24):
            start = admission - pd.Timedelta(hours=hours)
            base[f"volumen_ingresos_{hours}h"] = int(
                (ingress_dates.ge(start) & ingress_dates.lt(admission)).sum()
            )
        return base

    conduct_date = pd.to_datetime(patient["fecha_conducta"], errors="coerce")
    if pd.isna(conduct_date) or conduct_date < admission:
        raise ValueError("fecha_conducta debe ser válida y posterior al ingreso.")
    conduct_calendar = add_calendar_features(
        pd.DataFrame({"fecha_conducta": [conduct_date]}),
        "fecha_conducta",
        "conducta",
    ).iloc[0]
    for name in (
        "hora_conducta",
        "dia_semana_conducta",
        "mes_conducta",
        "fin_semana_conducta",
    ):
        base[name] = conduct_calendar[name]

    conducts = pd.to_datetime(context["fecha_conducta"], errors="coerce")
    exits = pd.to_datetime(context["fecha_salida"], errors="coerce")
    valid_ingress = ingress_dates.notna() & conducts.notna() & conducts.ge(ingress_dates)
    valid_exit = conducts.notna() & exits.notna() & exits.ge(conducts)
    active = valid_ingress & ingress_dates.lt(conduct_date) & conducts.ge(conduct_date)
    pending = valid_exit & conducts.lt(conduct_date) & exits.ge(conduct_date)
    base.update(
        {
            "pacientes_activos_en_ingreso": int(active.sum()),
            "pacientes_pendientes_de_salida": int(pending.sum()),
            "duracion_ingreso_calculada": float(
                (conduct_date - admission).total_seconds() / 3600
            ),
            "sala_observacion": patient["sala_observacion"],
            "grupo_cama": group_bed_category(patient["cama"]),
            "conducta": patient["conducta"],
        }
    )
    return base


def predict_new_patient(
    model_name: str,
    patient: dict[str, Any],
    context: pd.DataFrame,
    models: dict[str, Pipeline],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Predice un paciente nuevo con entradas observadas y variables derivadas."""
    predictors = build_new_patient_predictors(context, model_name, patient)
    result = predict_patient(model_name, predictors, models, metadata)
    automatic_names = {
        "volumen_ingresos_3h",
        "volumen_ingresos_6h",
        "volumen_ingresos_24h",
        "hora_ingreso",
        "dia_semana_ingreso",
        "mes_ingreso",
        "fin_semana_ingreso",
        "pacientes_activos_en_ingreso",
        "pacientes_pendientes_de_salida",
        "duracion_ingreso_calculada",
        "grupo_cama",
        "hora_conducta",
        "dia_semana_conducta",
        "mes_conducta",
        "fin_semana_conducta",
    }
    result["automatic_features"] = {
        name: (value.item() if hasattr(value, "item") else value)
        for name, value in predictors.items()
        if name in automatic_names
    }
    result["observed_features"] = {
        name: value for name, value in patient.items() if name in required_observed(model_name)
    }
    return result


def required_observed(model_name: str) -> set[str]:
    """Enumera los campos observados que puede editar el usuario por momento."""
    fields = {"edad", "triage", "afiliacion", "fecha_ingreso"}
    if model_name == "B":
        fields.update({"fecha_conducta", "conducta", "sala_observacion", "cama"})
    return fields
