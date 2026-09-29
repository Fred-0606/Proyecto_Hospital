"""Carga y valida la configuración congelada antes de abrir la prueba final."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.data_io import require_file
from src.paths import CONFIG_DIR, PROJECT_ROOT
from src.preprocessing import (
    MODEL_A_PREDICTORS,
    MODEL_A_TARGET,
    MODEL_B_PREDICTORS,
    MODEL_B_TARGET,
    TEST_START_DATE,
)

MODEL_FREEZE_PATH = CONFIG_DIR / "modelos_congelados.yml"
SELECTION_PATH = PROJECT_ROOT / "outputs" / "metrics" / "08_seleccion_modelos.json"


def _sha256(path: Path) -> str:
    """Calcula la huella SHA-256 de un archivo sin modificarlo."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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

    for relative_path, expected_hash in configuration["huellas_sha256"].items():
        artifact = PROJECT_ROOT / relative_path
        require_file(artifact, f"Falta el artefacto congelado: {relative_path}.")
        if _sha256(artifact) != expected_hash:
            raise ValueError(f"La huella cambió después del congelamiento: {relative_path}.")
    return configuration
