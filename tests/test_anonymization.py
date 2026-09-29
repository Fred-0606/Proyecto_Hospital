"""Pruebas de anonimización de identificadores directos en la etapa 01."""

import pandas as pd

from src.anonymization import anonymize_direct_identifiers, mask_value


def test_mask_value_preserves_only_requested_edges() -> None:
    """El enmascaramiento conserva dos caracteres iniciales y tres finales."""
    assert mask_value("usuario@hospital.org") == "us***************org"
    assert mask_value(None) is None


def test_anonymization_is_stable_within_dataset_and_preserves_analytics() -> None:
    """Valores repetidos mantienen seudónimo y las variables analíticas no cambian."""
    source = pd.DataFrame(
        {
            "identificacion": [123, 123, None],
            "nombre_paciente": ["Ana", "Ana", None],
            "user": ["ab@example.org", "cd@example.org", None],
            "no_modificar": ["xy@hospital.org", None, "zz@hospital.org"],
            "no_modificar_1": [6.5, 2.0, 8.0],
            "edad": [20, 30, 40],
        }
    )
    preserved = source.copy(deep=True)

    result, audit = anonymize_direct_identifiers(source, salt=b"salt-prueba-1234")

    assert result.loc[0, "identificacion"] == result.loc[1, "identificacion"]
    assert len(result.loc[0, "identificacion"]) == 64
    assert result.loc[0, "nombre_paciente"] == result.loc[1, "nombre_paciente"]
    assert len(result.loc[0, "nombre_paciente"]) == 32
    assert result.loc[0, "user"] == "ab*********org"
    assert result.loc[0, "no_modificar"] == "xy**********org"
    assert result["no_modificar_1"].tolist() == source["no_modificar_1"].tolist()
    assert result["edad"].tolist() == source["edad"].tolist()
    assert set(audit["columna"]) == {
        "identificacion",
        "nombre_paciente",
        "user",
        "no_modificar",
    }
    pd.testing.assert_frame_equal(source, preserved)
