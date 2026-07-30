"""Comprueba que las rutas del proyecto sean portables e independientes del CWD."""

from pathlib import Path

from src import paths


def test_get_project_root_matches_repository_location() -> None:
    """La raíz debe calcularse desde el módulo y no desde el proceso actual."""
    expected_root = Path(__file__).resolve().parents[1]

    assert paths.get_project_root() == expected_root


def test_directory_constants_are_relative_to_project_root() -> None:
    """Las constantes deben apuntar a la jerarquía esperada del repositorio."""
    root = paths.get_project_root()

    assert paths.CONFIG_DIR == root / "config"
    assert paths.DATA_DIR == root / "data"
    assert paths.RAW_DATA_DIR == root / "data" / "raw"
    assert paths.INTERIM_DATA_DIR == root / "data" / "interim"
    assert paths.PROCESSED_DATA_DIR == root / "data" / "processed"
    assert paths.OUTPUTS_DIR == root / "outputs"
    assert paths.FIGURES_DIR == root / "outputs" / "figures"
    assert paths.TABLES_DIR == root / "outputs" / "tables"
    assert paths.METRICS_DIR == root / "outputs" / "metrics"
    assert paths.MODELS_DIR == root / "outputs" / "models"


def test_output_directories_exist() -> None:
    """Los directorios destinados a resultados deben estar disponibles."""
    output_directories = (
        paths.OUTPUTS_DIR,
        paths.FIGURES_DIR,
        paths.TABLES_DIR,
        paths.METRICS_DIR,
        paths.MODELS_DIR,
    )

    assert all(directory.is_dir() for directory in output_directories)


def test_project_root_does_not_depend_on_working_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Cambiar el directorio de trabajo no debe alterar la raíz detectada."""
    expected_root = paths.get_project_root()
    monkeypatch.chdir(tmp_path)

    assert paths.get_project_root() == expected_root
