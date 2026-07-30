# Proyecto Hospital

## Descripción

Proyecto de grado de Maestría en Ciencia de Datos orientado al análisis y la predicción de estancias prolongadas en un servicio de urgencias hospitalarias de Bogotá. El proyecto utiliza Python, scikit-learn y Quarto, con una organización modular y un pipeline secuencial.

## Objetivo

Analizar el tiempo total entre el triage y la salida física, desarrollar modelos para predecirlo y construir una variable binaria de estancia prolongada. El umbral inicial es de 12 horas y se administra mediante `config/settings.yml`.

## Estructura

- `config/`: configuración reproducible del proyecto.
- `data/`: datos locales crudos, intermedios y procesados; no se versionan.
- `pipeline/`: documentos Quarto ordenados según las etapas del análisis.
- `src/`: código Python reutilizable.
- `scripts/`: puntos de entrada para validación y ejecución.
- `outputs/`: figuras, tablas, métricas, modelos y otros artefactos.
- `tests/`: pruebas básicas con pytest.

## Instalación en Windows

Se requieren Python y Quarto instalados.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Cuando el archivo de bloqueo contenga versiones validadas, se podrá reconstruir el entorno con:

```powershell
python -m pip install -r requirements-lock.txt
```

## Ejecución secuencial

Cada etapa debe ejecutarse en una sesión limpia y consumir el artefacto generado por la etapa anterior. El orden es:

1. Contexto y diccionario.
2. Ingesta y validación.
3. Limpieza y calidad.
4. Análisis exploratorio.
5. Ingeniería de variables.
6. Preprocesamiento.
7. Modelo baseline.
8. Modelos comparativos.
9. Evaluación.
10. Interpretabilidad.
11. Conclusiones.

Una etapa individual podrá ejecutarse mediante `scripts/run_step.py` cuando dicho script sea implementado. La ejecución completa se realizará mediante `scripts/run_pipeline.ps1`.

## Renderizado con Quarto

Para renderizar localmente el sitio completo en el orden definido en `_quarto.yml`:

```powershell
quarto render
```

El sitio se genera en `_site/`. Para previsualizarlo durante el desarrollo:

```powershell
quarto preview
```

El proyecto no configura publicación automática ni GitHub Pages.

## Política de datos

Los datasets, datos hospitalarios, credenciales, tokens y archivos `.env` nunca deben incluirse en Git. Esta restricción también aplica a datos con nombres ficticios, pues podrían conservar información clínica o cuasi-identificadores.

Los directorios `data/raw/`, `data/interim/` y `data/processed/` se mantienen únicamente mediante archivos `.gitkeep`. El archivo `database_modificado.xlsx` está bloqueado explícitamente. Solo deben versionarse tablas o figuras pequeñas, agregadas y necesarias para documentar resultados.

## Estado

El repositorio contiene la estructura base. El análisis todavía no ha sido implementado.
