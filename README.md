# Proyecto Hospital

## Descripción

Proyecto de grado de Maestría en Ciencia de Datos orientado al análisis y la predicción de estancias prolongadas en un servicio de urgencias hospitalarias de Bogotá. El proyecto utiliza Python, scikit-learn y Quarto, con una organización modular y un pipeline secuencial.

## Objetivo

Desarrollar dos modelos de clasificación: uno para predecir si el tiempo entre ingreso y conducta es mayor o igual a seis horas, y otro para predecir si el tiempo entre conducta y egreso administrativo es mayor o igual a seis horas. El umbral corresponde a un objetivo institucional y se administra mediante `config/settings.yml`.

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

1. Contexto y objetivos.
2. ETL de extracción y anonimización.
3. Diseño analítico y selección de variables.
4. ETL de transformación y calidad.
5. Construcción de los conjuntos de datos y partición temporal.
6. EDA únicamente sobre desarrollo.
7. Feature engineering y preprocesamiento.
8. Modelos baseline.
9. Modelos comparativos.
10. Evaluación final.
11. Interpretabilidad.
12. Conclusiones.

La etapa 06 documenta las variables temporales y operativas derivadas. Para el
Modelo A incluye el volumen de ingresos estrictamente anteriores durante las
ventanas de 3, 6 y 24 horas, calculado sin utilizar la atención actual ni
eventos futuros.

Una etapa individual se ejecuta indicando su número, por ejemplo:

```powershell
.\.venv\Scripts\python.exe scripts\run_step.py 05
```

La ejecución completa se realiza mediante:

```powershell
.\scripts\run_pipeline.ps1
```

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

Los directorios `data/raw/`, `data/interim/` y `data/processed/` se mantienen únicamente mediante archivos `.gitkeep`. Los libros `BD_CENSO_2024_2026.xlsx` y `database_modificado.xlsx` están bloqueados explícitamente. Solo deben versionarse tablas o figuras pequeñas, agregadas y necesarias para documentar resultados.

La etapa 01 utiliza `data/raw/BD_CENSO_2024_2026.xlsx` y anonimiza los identificadores directos antes de escribir el primer Parquet: aplica hash SHA-256 con salt efímero a la identificación, tokens UUID al nombre y enmascaramiento a las columnas operativas que contienen correos. El salt y el mapa de tokens no se persisten.

## Estado

Las etapas 00 a 08 están terminadas. Las etapas 09 a 11 permanecen pendientes:
evaluación final sobre la prueba temporal reservada, interpretabilidad y
conclusiones.

La configuración previa a la evaluación final está congelada en
`config/modelos_congelados.yml`. La etapa 09 debe validar ese manifiesto y sus
huellas antes de abrir las particiones de prueba.
