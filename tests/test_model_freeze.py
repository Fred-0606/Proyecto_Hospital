"""Pruebas del contrato congelado previo a la evaluación final."""

from pathlib import Path

import pytest

import src.model_freeze as model_freeze
from src.model_freeze import (
    build_frozen_pipeline,
    create_test_opening_record,
    validate_frozen_configuration,
)


def test_frozen_configuration_matches_current_pipeline() -> None:
    """Predictores, selección y artefactos deben coincidir con el manifiesto."""
    configuration = validate_frozen_configuration()

    assert configuration["estado"] == "congelado_previo_a_prueba"
    assert configuration["prueba_final_abierta"] is False
    assert len(configuration["commit_metodologico"]) == 40
    assert set(configuration["huellas_particiones"]) == {
        "modelo_a_desarrollo",
        "modelo_a_prueba",
        "modelo_b_desarrollo",
        "modelo_b_prueba",
    }
    assert configuration["modelos"]["A"]["candidato"] == "regresion_logistica"
    assert configuration["modelos"]["B"]["candidato"] == "regresion_logistica"


@pytest.mark.parametrize("model_name", ["A", "B"])
def test_frozen_pipeline_uses_every_classifier_parameter(model_name: str) -> None:
    """El constructor final debe aplicar todos los parámetros del manifiesto."""
    configuration = validate_frozen_configuration()
    classifier_parameters = build_frozen_pipeline(
        model_name,
        configuration,
    )["clasificador"].get_params()

    for parameter, expected_value in configuration["modelos"][model_name][
        "clasificador"
    ].items():
        assert classifier_parameters[parameter] == expected_value


def test_opening_requires_confirmation_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El registro de apertura exige confirmación y se crea una sola vez."""
    output_path = tmp_path / "09_apertura_prueba.json"
    with pytest.raises(ValueError, match="confirm_opening=True"):
        create_test_opening_record(output_path=output_path)

    def fake_git_output(*arguments: str) -> str:
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        return ""

    monkeypatch.setattr(model_freeze, "_git_output", fake_git_output)
    record = create_test_opening_record(
        confirm_opening=True,
        output_path=output_path,
    )
    assert record["estado"] == "prueba_final_abierta"
    assert output_path.is_file()

    with pytest.raises(FileExistsError, match="ya tiene un registro"):
        create_test_opening_record(
            confirm_opening=True,
            output_path=output_path,
        )
