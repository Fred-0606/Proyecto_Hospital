"""Carga y valida la configuración congelada antes de abrir la prueba final."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_io import require_file
from src.paths import CONFIG_DIR, METRICS_DIR, PROJECT_ROOT
from src.preprocessing import (
    MODEL_A_DEVELOPMENT_PATH,
    MODEL_A_PREDICTORS,
    MODEL_A_TEST_PATH,
    MODEL_A_TARGET,
    MODEL_B_DEVELOPMENT_PATH,
    MODEL_B_PREDICTORS,
    MODEL_B_TEST_PATH,
    MODEL_B_TARGET,
    TEST_START_DATE,
    VALIDATION_PERIODS,
    build_model_a_preprocessor,
    build_model_b_preprocessor,
)

MODEL_FREEZE_PATH = CONFIG_DIR / "modelos_congelados.yml"
SELECTION_PATH = PROJECT_ROOT / "outputs" / "metrics" / "08_seleccion_modelos.json"
TEST_OPENING_RECORD_PATH = METRICS_DIR / "09_apertura_prueba.json"
RANDOM_STATE = 42
SELECTION_METRIC = "pr_auc"

PARTITION_PATHS = {
    "modelo_a_desarrollo": MODEL_A_DEVELOPMENT_PATH,
    "modelo_a_prueba": MODEL_A_TEST_PATH,
    "modelo_b_desarrollo": MODEL_B_DEVELOPMENT_PATH,
    "modelo_b_prueba": MODEL_B_TEST_PATH,
}


def _sha256(path: Path) -> str:
    """Calcula la huella SHA-256 de un archivo sin modificarlo."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    """Ejecuta una consulta Git de solo lectura desde la raíz del proyecto."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_frozen_configuration(path: Path = MODEL_FREEZE_PATH) -> dict[str, Any]:
    """Lee el manifiesto congelado y exige una estructura de diccionario."""
    require_file(path, "Congele primero la configuración previa a la etapa 09.")
    with path.open(encoding="utf-8") as file:
        configuration = yaml.safe_load(file)
    if not isinstance(configuration, dict):
        raise ValueError("La configuración congelada debe ser un diccionario YAML.")
    return configuration


def validate_frozen_configuration(
    path: Path = MODEL_FREEZE_PATH,
) -> dict[str, Any]:
    """Comprueba predictores, selección y huellas antes de usar la prueba."""
    configuration = load_frozen_configuration(path)
    if configuration.get("prueba_final_abierta") is not False:
        raise ValueError("El manifiesto previo a prueba debe mantener la prueba cerrada.")
    if str(configuration.get("fecha_inicio_prueba")) != str(TEST_START_DATE.date()):
        raise ValueError("La fecha congelada de inicio de prueba no coincide con el código.")
    if configuration.get("random_state") != RANDOM_STATE:
        raise ValueError("La semilla congelada no coincide con la semilla oficial.")
    if configuration.get("metrica_seleccion") != SELECTION_METRIC:
        raise ValueError("La métrica congelada de selección no coincide con el código.")
    if configuration.get("pliegues_temporales") != len(VALIDATION_PERIODS):
        raise ValueError("Cambió la cantidad de pliegues temporales congelados.")

    frozen_commit = str(configuration.get("commit_metodologico", ""))
    if not frozen_commit:
        raise ValueError("Falta registrar el commit metodológico del congelamiento.")
    try:
        _git_output("cat-file", "-e", f"{frozen_commit}^{{commit}}")
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError("El commit metodológico congelado no existe en Git.") from error

    expected_models = {
        "A": (MODEL_A_TARGET, MODEL_A_PREDICTORS),
        "B": (MODEL_B_TARGET, MODEL_B_PREDICTORS),
    }
    for model_name, (target, predictors) in expected_models.items():
        frozen = configuration["modelos"][model_name]
        if frozen["objetivo"] != target or frozen["predictores"] != predictors:
            raise ValueError(
                f"El contrato congelado del Modelo {model_name} no coincide con el código."
            )

    require_file(SELECTION_PATH, "Ejecute primero la etapa 08 de selección.")
    with SELECTION_PATH.open(encoding="utf-8") as file:
        selection = json.load(file)["seleccion"]
    for model_name in expected_models:
        frozen = configuration["modelos"][model_name]
        selected = selection[model_name]
        if frozen["candidato"] != selected["candidato"]:
            raise ValueError(f"Cambió el candidato seleccionado para el Modelo {model_name}.")
        if float(frozen["umbral"]) != float(selected["umbral"]):
            raise ValueError(f"Cambió el umbral seleccionado para el Modelo {model_name}.")
        selected_parameters = selected["hiperparametros"]
        for parameter in ("C", "class_weight", "l1_ratio"):
            if frozen["clasificador"][parameter] != selected_parameters[
                f"clasificador__{parameter}"
            ]:
                raise ValueError(
                    f"Cambió {parameter} en el Modelo {model_name} después del congelamiento."
                )

        classifier = LogisticRegression(**frozen["clasificador"])
        effective_parameters = classifier.get_params()
        for parameter, expected_value in frozen["clasificador"].items():
            if effective_parameters[parameter] != expected_value:
                raise ValueError(
                    f"El clasificador no reconoce el valor congelado de {parameter} "
                    f"para el Modelo {model_name}."
                )

    frozen_partitions = configuration.get("huellas_particiones", {})
    for name, partition_path in PARTITION_PATHS.items():
        require_file(partition_path, f"Falta la partición congelada: {name}.")
        if frozen_partitions.get(name) != _sha256(partition_path):
            raise ValueError(f"Cambió la partición congelada: {name}.")

    for relative_path, expected_hash in configuration["huellas_sha256"].items():
        artifact = PROJECT_ROOT / relative_path
        require_file(artifact, f"Falta el artefacto congelado: {relative_path}.")
        if _sha256(artifact) != expected_hash:
            raise ValueError(f"La huella cambió después del congelamiento: {relative_path}.")
    return configuration


def build_frozen_pipeline(
    model_name: str,
    configuration: dict[str, Any] | None = None,
) -> Pipeline:
    """Construye exactamente el pipeline registrado para el Modelo A o B."""
    frozen = configuration or validate_frozen_configuration()
    if model_name not in {"A", "B"}:
        raise ValueError("model_name debe ser 'A' o 'B'.")
    preprocessor_builder = (
        build_model_a_preprocessor if model_name == "A" else build_model_b_preprocessor
    )
    classifier = LogisticRegression(**frozen["modelos"][model_name]["clasificador"])
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocessor_builder()),
            ("clasificador", classifier),
        ]
    )


def create_test_opening_record(
    *,
    confirm_opening: bool = False,
    output_path: Path = TEST_OPENING_RECORD_PATH,
) -> dict[str, Any]:
    """Registra una apertura única sin modificar el manifiesto congelado.

    La creación usa modo exclusivo: si el registro ya existe, no puede
    sobrescribirse accidentalmente. Esta función debe invocarse inmediatamente
    antes de cargar las particiones de prueba en la etapa 09.
    """
    if not confirm_opening:
        raise ValueError("La apertura requiere confirm_opening=True de forma explícita.")
    configuration = validate_frozen_configuration()
    if output_path.exists():
        raise FileExistsError(
            f"La prueba ya tiene un registro de apertura: {output_path.name}."
        )
    if _git_output("status", "--porcelain"):
        raise RuntimeError("Git debe estar limpio antes de abrir la prueba final.")

    record = {
        "estado": "prueba_final_abierta",
        "fecha_hora_utc": datetime.now(timezone.utc).isoformat(),
        "confirmacion_explicita": True,
        "manifiesto": str(MODEL_FREEZE_PATH.relative_to(PROJECT_ROOT)),
        "sha256_manifiesto": _sha256(MODEL_FREEZE_PATH),
        "commit_metodologico": configuration["commit_metodologico"],
        "commit_ejecucion": _git_output("rev-parse", "HEAD"),
        "huellas_particiones": configuration["huellas_particiones"],
        "regla": "evaluacion_unica_sin_reajuste_posterior",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return record
