"""Pruebas del contrato congelado previo a la evaluación final."""

from src.model_freeze import validate_frozen_configuration


def test_frozen_configuration_matches_current_pipeline() -> None:
    """Predictores, selección y artefactos deben coincidir con el manifiesto."""
    configuration = validate_frozen_configuration()

    assert configuration["estado"] == "congelado_previo_a_prueba"
    assert configuration["prueba_final_abierta"] is False
    assert configuration["modelos"]["A"]["candidato"] == "regresion_logistica"
    assert configuration["modelos"]["B"]["candidato"] == "regresion_logistica"
