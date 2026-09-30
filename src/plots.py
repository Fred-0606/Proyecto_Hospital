"""Centraliza visualizaciones reproducibles y seguras para los informes del proyecto."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

NAVY = "#12304A"
BLUE = "#2F6690"
TEAL = "#3A7D7C"
CORAL = "#D97757"
LIGHT_BLUE = "#DDEBF4"
GRID = "#D5DEE6"


def _finish_figure(figure: plt.Figure, output_path: Path) -> Path:
    """Guarda una figura con parámetros homogéneos y libera memoria."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def _style_axis(axis: plt.Axes) -> None:
    """Aplica el estilo visual común del proyecto a un eje."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors="#334A5E")
    axis.title.set_color(NAVY)
    axis.xaxis.label.set_color(NAVY)
    axis.yaxis.label.set_color(NAVY)
    axis.set_axisbelow(True)


def plot_null_counts(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Grafica y guarda la cantidad de valores nulos de cada columna."""
    null_counts = dataframe.isna().sum().sort_values(ascending=True)
    figure_height = max(6.0, len(null_counts) * 0.36)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    bars = axis.barh(null_counts.index.astype(str), null_counts.values)
    maximum_count = max(int(null_counts.max()), 1)

    axis.bar_label(bars, labels=[f"{int(value):,}" for value in null_counts.values], padding=3)
    axis.set_title("Cantidad de valores nulos por columna — dataset resultante")
    axis.set_xlabel("Cantidad de valores nulos")
    axis.set_ylabel("Columna")
    axis.set_xlim(0, maximum_count * 1.12)
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()

    return _finish_figure(figure, output_path)


def plot_duration_distributions(
    model_a: pd.DataFrame,
    model_b: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Compara las distribuciones de los tiempos A y B con escala legible."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    configurations = (
        (
            axes[0],
            model_a["duracion_ingreso_calculada"],
            "Tiempo A — ingreso",
            BLUE,
        ),
        (
            axes[1],
            model_b["duracion_salida_calculada"],
            "Tiempo B — salida",
            TEAL,
        ),
    )
    for axis, series, title, color in configurations:
        limit = float(series.quantile(0.99))
        visible = series.clip(upper=limit)
        axis.hist(visible, bins=35, color=color, alpha=0.9, edgecolor="white")
        axis.axvline(6, color=CORAL, linewidth=2, linestyle="--", label="Umbral: 6 h")
        axis.axvline(
            series.median(), color=NAVY, linewidth=1.8, label=f"Mediana: {series.median():.1f} h"
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Duración (horas; valores sobre P99 agrupados en el extremo)")
        axis.grid(axis="y", color=GRID, alpha=0.65)
        axis.legend(frameon=False, fontsize=9)
        _style_axis(axis)
    axes[0].set_ylabel("Número de registros")
    figure.suptitle(
        "Distribución de los tiempos hospitalarios",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish_figure(figure, output_path)


def plot_target_balance(balance: pd.DataFrame, output_path: Path) -> Path:
    """Muestra el balance de clases de ambos objetivos en paneles comparables."""
    models = balance["modelo"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(models), figsize=(11.5, 4.6), sharey=True)
    axes_array = np.atleast_1d(axes)
    for axis, model_name in zip(axes_array, models, strict=True):
        subset = balance.loc[balance["modelo"].eq(model_name)]
        bars = axis.bar(
            subset["etiqueta"], subset["porcentaje"], color=[LIGHT_BLUE, BLUE], width=0.62
        )
        axis.bar_label(
            bars,
            labels=[
                f"{percentage:.1f}%\n({records:,.0f})"
                for percentage, records in zip(
                    subset["porcentaje"], subset["registros"], strict=True
                )
            ],
            padding=4,
            color=NAVY,
            fontsize=10,
        )
        axis.set_title(model_name, loc="left", fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylim(0, max(70, float(balance["porcentaje"].max()) * 1.18))
        axis.grid(axis="y", color=GRID, alpha=0.65)
        _style_axis(axis)
    axes_array[0].set_ylabel("Porcentaje de registros")
    figure.suptitle(
        "Balance de las variables objetivo",
        x=0.07,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish_figure(figure, output_path)


def plot_target_rate_by_triage(rates: pd.DataFrame, output_path: Path) -> Path:
    """Compara la proporción prolongada por nivel de triage y modelo."""
    categories = sorted(rates["triage"].astype(str).unique())
    display_labels = [category.replace("_", " ").capitalize() for category in categories]
    models = rates["modelo"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(models), figsize=(13, 5), sharey=True)
    axes_array = np.atleast_1d(axes)
    for axis, model_name, color in zip(axes_array, models, [BLUE, TEAL], strict=True):
        subset = rates.loc[rates["modelo"].eq(model_name)].set_index("triage")
        percentages = subset.reindex(categories)["tasa_prolongada"].fillna(0)
        bars = axis.barh(display_labels, percentages, color=color, alpha=0.9)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in percentages], padding=3)
        axis.set_title(model_name, loc="left", fontweight="bold")
        axis.set_xlabel("Registros con estancia prolongada (%)")
        axis.grid(axis="x", color=GRID, alpha=0.65)
        _style_axis(axis)
    figure.suptitle(
        "Estancia prolongada según triage",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish_figure(figure, output_path)


def plot_target_rate_by_hour(rates: pd.DataFrame, output_path: Path) -> Path:
    """Compara la tasa prolongada según la hora de referencia de cada modelo."""
    figure, axis = plt.subplots(figsize=(12, 5.2))
    for model_name, color, marker in (
        ("Modelo A — ingreso", BLUE, "o"),
        ("Modelo B — salida", TEAL, "s"),
    ):
        subset = rates.loc[rates["modelo"].eq(model_name)].sort_values("hora")
        axis.plot(
            subset["hora"],
            subset["tasa_prolongada"],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=4.5,
            label=model_name,
        )
    axis.set_title("Estancia prolongada según la hora operativa", loc="left", fontsize=16, fontweight="bold")
    axis.set_xlabel("Hora del ingreso (A) o de la conducta (B)")
    axis.set_ylabel("Registros con estancia prolongada (%)")
    axis.set_xticks(range(0, 24, 2))
    axis.grid(color=GRID, alpha=0.65)
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)
    figure.tight_layout()
    return _finish_figure(figure, output_path)


def plot_baseline_confusion_matrices(
    validation_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Grafica matrices de confusión agregadas de los pliegues de validación."""
    models = validation_metrics["modelo"].drop_duplicates().tolist()
    baselines = validation_metrics["baseline"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(models),
        len(baselines),
        figsize=(6.2 * len(baselines), 5.2 * len(models)),
        squeeze=False,
    )
    display_names = {
        "clase_mayoritaria": "Clase mayoritaria",
        "regresion_logistica_l2": "Regresión logística L2",
    }
    count_columns = [
        "verdaderos_negativos",
        "falsos_positivos",
        "falsos_negativos",
        "verdaderos_positivos",
    ]
    maximum = float(validation_metrics[count_columns].sum(axis=1).max())

    for row, model_name in enumerate(models):
        for column, baseline_name in enumerate(baselines):
            subset = validation_metrics.loc[
                validation_metrics["modelo"].eq(model_name)
                & validation_metrics["baseline"].eq(baseline_name)
            ]
            matrix = np.array(
                [
                    [subset["verdaderos_negativos"].sum(), subset["falsos_positivos"].sum()],
                    [subset["falsos_negativos"].sum(), subset["verdaderos_positivos"].sum()],
                ]
            )
            axis = axes[row, column]
            axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
            for matrix_row in range(2):
                for matrix_column in range(2):
                    value = int(matrix[matrix_row, matrix_column])
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{value:,}",
                        ha="center",
                        va="center",
                        color="white" if value > maximum * 0.5 else NAVY,
                        fontsize=13,
                        fontweight="bold",
                    )
            axis.set_xticks([0, 1], labels=["Negativa", "Positiva"])
            axis.set_yticks([0, 1], labels=["Negativa", "Positiva"])
            axis.set_xlabel("Clase predicha")
            axis.set_ylabel("Clase real")
            axis.set_title(
                f"Modelo {model_name} — {display_names.get(baseline_name, baseline_name)}",
                fontweight="bold",
            )
    figure.suptitle(
        "Matrices de confusión acumuladas en validación temporal",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish_figure(figure, output_path)


def plot_baseline_metric_comparison(
    validation_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Compara las métricas medias de validación entre los baselines."""
    metrics = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]
    models = validation_summary["modelo"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1, len(models), figsize=(14, 5.4), sharey=True, squeeze=False
    )
    colors = {"clase_mayoritaria": LIGHT_BLUE, "regresion_logistica_l2": BLUE}
    display_names = {
        "clase_mayoritaria": "Clase mayoritaria",
        "regresion_logistica_l2": "Regresión logística L2",
    }
    x_positions = np.arange(len(metrics))
    width = 0.36

    for axis, model_name in zip(axes[0], models, strict=True):
        subset = validation_summary.loc[validation_summary["modelo"].eq(model_name)]
        for index, (_, row) in enumerate(subset.iterrows()):
            baseline_name = str(row["baseline"])
            values = [float(row[f"{metric}_mean"]) for metric in metrics]
            offset = (index - (len(subset) - 1) / 2) * width
            bars = axis.bar(
                x_positions + offset,
                values,
                width,
                color=colors.get(baseline_name, TEAL),
                label=display_names.get(baseline_name, baseline_name),
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=2,
                fontsize=8,
            )
        axis.set_title(f"Modelo {model_name}", loc="left", fontweight="bold")
        axis.set_xticks(x_positions, metric_labels, rotation=35, ha="right")
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", color=GRID, alpha=0.65)
        axis.legend(frameon=False, fontsize=9)
        _style_axis(axis)
    axes[0, 0].set_ylabel("Métrica media en validación")
    figure.suptitle(
        "Comparación de modelos baseline",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish_figure(figure, output_path)


def plot_candidate_metric_comparison(
    validation_summary: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Compara las métricas medias de validación de cuatro candidatos."""
    metrics = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]
    candidate_names = {
        "regresion_logistica": "Regresión logística",
        "arbol_decision": "Árbol de decisión",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    candidate_colors = [BLUE, CORAL, TEAL, NAVY]
    models = validation_summary["modelo"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1, len(models), figsize=(15, 5.8), sharey=True, squeeze=False
    )
    x_positions = np.arange(len(metrics))
    width = 0.18

    for axis, model_name in zip(axes[0], models, strict=True):
        subset = validation_summary.loc[validation_summary["modelo"].eq(model_name)]
        for index, (_, row) in enumerate(subset.iterrows()):
            candidate = str(row["candidato"])
            values = [float(row[f"{metric}_mean"]) for metric in metrics]
            offset = (index - (len(subset) - 1) / 2) * width
            axis.bar(
                x_positions + offset,
                values,
                width,
                color=candidate_colors[index],
                label=candidate_names.get(candidate, candidate),
            )
        axis.set_title(f"Modelo {model_name}", loc="left", fontweight="bold")
        axis.set_xticks(x_positions, metric_labels)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color=GRID, alpha=0.65)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        _style_axis(axis)
    axes[0, 0].set_ylabel("Métrica media en validación")
    figure.suptitle(
        "Comparación temporal de modelos candidatos",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish_figure(figure, output_path)


def plot_selected_confusion_matrices(
    selected_metrics: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Grafica una matriz de confusión para el candidato elegido de cada tarea."""
    figure, axes = plt.subplots(1, len(selected_metrics), figsize=(12, 4.8), squeeze=False)
    maximum = float(
        selected_metrics[
            ["verdaderos_negativos", "falsos_positivos", "falsos_negativos", "verdaderos_positivos"]
        ].to_numpy().max()
    )
    display_names = {
        "regresion_logistica": "Regresión logística",
        "arbol_decision": "Árbol de decisión",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    for axis, (_, row) in zip(axes[0], selected_metrics.iterrows(), strict=True):
        matrix = np.array(
            [
                [row["verdaderos_negativos"], row["falsos_positivos"]],
                [row["falsos_negativos"], row["verdaderos_positivos"]],
            ],
            dtype=int,
        )
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for matrix_row in range(2):
            for matrix_column in range(2):
                value = int(matrix[matrix_row, matrix_column])
                axis.text(
                    matrix_column,
                    matrix_row,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    color="white" if value > maximum * 0.5 else NAVY,
                    fontsize=13,
                    fontweight="bold",
                )
        axis.set_xticks([0, 1], labels=["Negativa", "Positiva"])
        axis.set_yticks([0, 1], labels=["Negativa", "Positiva"])
        axis.set_xlabel("Clase predicha")
        axis.set_ylabel("Clase real")
        axis.set_title(
            f"Modelo {row['modelo']} — {display_names.get(row['candidato'], row['candidato'])}"
            f"\nUmbral: {row['umbral']:.3f}",
            fontweight="bold",
        )
    figure.suptitle(
        "Matrices de confusión de los modelos seleccionados",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    return _finish_figure(figure, output_path)
