"""Prueba la ingesta de CENSO sin utilizar el archivo hospitalario."""

from pathlib import Path

import pandas as pd
import pytest

from src import data_io


class FakeExcelFile:
    """Simula únicamente el acceso a nombres de hojas de un libro."""

    def __init__(self, sheet_names: list[str]) -> None:
        """Inicializa la lista simulada de hojas."""
        self.sheet_names = sheet_names

    def __enter__(self) -> "FakeExcelFile":
        """Abre el contexto simulado."""
        return self

    def __exit__(self, *args: object) -> None:
        """Cierra el contexto simulado."""


def test_get_sheet_names_without_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La lista de hojas debe obtenerse sin invocar read_excel."""
    source = tmp_path / "book.xlsx"
    source.touch()
    monkeypatch.setattr(
        data_io.pd,
        "ExcelFile",
        lambda _: FakeExcelFile(["HOME", "CENSO"]),
    )
    monkeypatch.setattr(
        data_io.pd,
        "read_excel",
        lambda *args, **kwargs: pytest.fail("No se debe cargar una hoja."),
    )

    assert data_io.get_excel_sheet_names(source) == ["HOME", "CENSO"]


def test_require_sheet_reports_missing_censo(tmp_path: Path) -> None:
    """La ausencia de CENSO debe informar archivo, hojas y hoja requerida."""
    source = tmp_path / "book.xlsx"

    with pytest.raises(ValueError, match="hoja requerida: CENSO"):
        data_io.require_sheet(source, ["HOME", "SALAS"])


def test_read_censo_uses_only_required_sheet_and_second_row_as_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """La lectura simulada debe solicitar CENSO con header=1."""
    source = tmp_path / "book.xlsx"
    source.touch()
    expected = pd.DataFrame({"A": [1]})
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        data_io,
        "get_excel_sheet_names",
        lambda _: ["HOME", "CENSO", "SALAS"],
    )

    def fake_read_excel(path: Path, **kwargs: object) -> pd.DataFrame:
        calls.append({"path": path, **kwargs})
        return expected

    monkeypatch.setattr(data_io.pd, "read_excel", fake_read_excel)

    result = data_io.read_censo_sheet(source)

    pd.testing.assert_frame_equal(result, expected)
    assert calls == [{"path": source, "sheet_name": "CENSO", "header": 1}]


def test_normalize_columns_and_resolve_duplicates() -> None:
    """Los nombres equivalentes deben recibir sufijos únicos y conservar el mapa."""
    dataframe = pd.DataFrame([[1, 2, 3]], columns=["  Fecha Á", "Fecha A", "Símbolo #%"])

    normalized, mapping = data_io.normalize_dataframe_columns(dataframe)

    assert list(normalized.columns) == ["fecha_a", "fecha_a__2", "simbolo"]
    assert mapping == {
        "fecha_a": "  Fecha Á",
        "fecha_a__2": "Fecha A",
        "simbolo": "Símbolo #%",
    }
    assert normalized.iloc[0].tolist() == [1, 2, 3]


def test_copy_dataframe_is_independent() -> None:
    """La copia debe poder cambiar sin alterar el DataFrame original."""
    original = pd.DataFrame({"valor": [1]})
    copied = data_io.copy_dataframe(original)
    copied.loc[0, "valor"] = 2

    assert original.loc[0, "valor"] == 1


def test_write_and_read_parquet(tmp_path: Path) -> None:
    """El artefacto Parquet debe conservar valores y columnas."""
    dataframe = pd.DataFrame({"valor": [1, 2], "texto": ["a", None]})
    output = tmp_path / "result.parquet"

    data_io.save_dataframe_as_parquet(dataframe, output)

    pd.testing.assert_frame_equal(pd.read_parquet(output), dataframe)


def test_parquet_serializes_mixed_object_column_without_mutating_source(
    tmp_path: Path,
) -> None:
    """Una columna mixta debe serializarse como texto sin mutar el origen."""
    dataframe = pd.DataFrame({"triage": ["I", 2, None]})
    preserved = dataframe.copy(deep=True)
    output = tmp_path / "mixed.parquet"

    data_io.save_dataframe_as_parquet(dataframe, output)
    restored = pd.read_parquet(output)

    pd.testing.assert_frame_equal(dataframe, preserved)
    assert restored["triage"].iloc[:2].tolist() == ["I", "2"]
    assert pd.isna(restored["triage"].iloc[2])


def test_save_json(tmp_path: Path) -> None:
    """Los metadatos deben guardarse como JSON legible."""
    output = tmp_path / "metadata.json"

    data_io.save_json({"filas": 2}, output)

    assert output.read_text(encoding="utf-8").strip().startswith("{")
    assert '"filas": 2' in output.read_text(encoding="utf-8")
