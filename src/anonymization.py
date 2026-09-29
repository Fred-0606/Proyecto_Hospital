"""Anonimiza identificadores directos antes de guardar artefactos derivados."""

from __future__ import annotations

import hashlib
import os
import uuid

import pandas as pd


def hash_with_salt(value: object, salt: bytes) -> object:
    """Devuelve un SHA-256 con salt o conserva un valor nulo."""
    if pd.isna(value):
        return None
    return hashlib.sha256(salt + str(value).encode("utf-8")).hexdigest()


def tokenize_series(series: pd.Series) -> pd.Series:
    """Asigna un token UUID estable por nombre dentro de la ejecución."""
    token_map: dict[object, str] = {}

    def to_token(value: object) -> object:
        if pd.isna(value):
            return None
        if value not in token_map:
            token_map[value] = uuid.uuid4().hex
        return token_map[value]

    return series.map(to_token).astype("string")


def mask_value(value: object) -> object:
    """Oculta el interior y conserva los dos primeros y tres últimos caracteres."""
    if pd.isna(value):
        return None
    text = str(value)
    if len(text) <= 5:
        return "*" * len(text)
    return text[:2] + "*" * (len(text) - 5) + text[-3:]


def anonymize_direct_identifiers(
    dataframe: pd.DataFrame,
    salt: bytes | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anonimiza identificación, nombre y columnas operativas con correos."""
    anonymized = dataframe.copy(deep=True)
    execution_salt = salt or os.urandom(16)
    audit_rows: list[dict[str, object]] = []

    transformations = {
        "identificacion": (
            "hash_sha256_con_salt_efimero",
            lambda series: series.map(
                lambda value: hash_with_salt(value, execution_salt)
            ).astype("string"),
        ),
        "nombre_paciente": ("token_uuid", tokenize_series),
    }

    email_columns = [
        column
        for column in anonymized.columns
        if column.startswith(("user", "no_modificar"))
        and anonymized[column].astype("string").str.contains("@", regex=False, na=False).any()
    ]
    transformations.update(
        {column: ("enmascaramiento_2_inicio_3_final", lambda series: series.map(mask_value).astype("string")) for column in email_columns}
    )

    for column, (method, transformation) in transformations.items():
        if column not in anonymized.columns:
            raise ValueError(f"Falta la columna sensible requerida: {column}")
        original = anonymized[column].copy()
        anonymized[column] = transformation(original)
        audit_rows.append(
            {
                "columna": column,
                "metodo": method,
                "valores_no_nulos": int(original.notna().sum()),
                "valores_transformados": int(
                    (original.astype("string") != anonymized[column]).fillna(False).sum()
                ),
            }
        )

    return anonymized, pd.DataFrame(audit_rows)
