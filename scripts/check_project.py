"""Verifica que la infraestructura mínima del proyecto esté disponible."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRECTORIES = (
    "config",
    "data",
    "data/raw",
    "data/interim",
    "data/processed",
    "pipeline",
    "src",
    "scripts",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/metrics",
    "outputs/models",
    "outputs/reports",
    "tests",
)

REQUIRED_DEPENDENCIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pyarrow": "pyarrow",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "PyYAML": "yaml",
    "joblib": "joblib",
    "Jupyter": "jupyter",
    "ipykernel": "ipykernel",
    "pytest": "pytest",
    "pandera": "pandera",
}


def check_directories() -> list[str]:
    """Devuelve mensajes de error para las carpetas requeridas ausentes."""
    return [
        f"Falta la carpeta requerida: {relative_path}"
        for relative_path in REQUIRED_DIRECTORIES
        if not (PROJECT_ROOT / relative_path).is_dir()
    ]


def check_virtual_environment() -> list[str]:
    """Comprueba la existencia del entorno virtual local."""
    if (PROJECT_ROOT / ".venv").is_dir():
        return []
    return ["Falta el entorno virtual .venv."]


def check_configuration() -> list[str]:
    """Comprueba la disponibilidad del archivo de configuración principal."""
    if (PROJECT_ROOT / "config" / "settings.yml").is_file():
        return []
    return ["Falta config/settings.yml."]


def check_src_import() -> list[str]:
    """Comprueba que el paquete src pueda importarse desde la raíz detectada."""
    project_root_text = str(PROJECT_ROOT)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    try:
        importlib.import_module("src")
    except ImportError as error:
        return [f"No fue posible importar src: {error}"]
    return []


def check_dependencies() -> list[str]:
    """Identifica las dependencias principales que no están instaladas."""
    return [
        f"Dependencia no disponible: {distribution_name}"
        for distribution_name, import_name in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(import_name) is None
    ]


def check_quarto() -> list[str]:
    """Comprueba Quarto con una invocación sin shell y tiempo limitado."""
    quarto_executable = shutil.which("quarto")
    if quarto_executable is None:
        return ["Quarto no está disponible en PATH."]

    try:
        result = subprocess.run(
            [quarto_executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return [f"No fue posible ejecutar Quarto de forma segura: {error}"]

    if result.returncode != 0:
        return ["Quarto está instalado, pero `quarto --version` falló."]
    return []


def run_checks() -> list[str]:
    """Ejecuta todas las verificaciones y devuelve los problemas encontrados."""
    checks = (
        check_directories,
        check_virtual_environment,
        check_configuration,
        check_src_import,
        check_dependencies,
        check_quarto,
    )
    errors: list[str] = []
    for check in checks:
        errors.extend(check())
    return errors


def main() -> int:
    """Muestra el resultado de las verificaciones y devuelve un código de salida."""
    errors = run_checks()
    if errors:
        print("Verificación del proyecto: INCOMPLETA")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Verificación del proyecto: CORRECTA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
