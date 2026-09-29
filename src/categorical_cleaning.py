"""Homologa categorías clínicas y administrativas seleccionadas."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

SPECIALTY_RULES: tuple[tuple[str, str], ...] = (
    (r"NEURO\s*(?:CX|QX)|NEUROCIR", "neurocirugia"),
    (r"CIRUGIA CARDIOVASCULAR", "cirugia_cardiovascular"),
    (r"CIRUGIA VASCULAR|(?:CX|QX) VASCULAR", "cirugia_vascular"),
    (r"CIRUGIA PEDIATR|(?:CX|QX) PEDIATR", "cirugia_pediatrica"),
    (r"CIRUGIA (?:DE )?TORAX", "cirugia_torax"),
    (r"MAXILO\s*FACIAL", "cirugia_maxilofacial"),
    (r"CIRUGIA PLASTICA|(?:CX|QX) PLASTICA|^PLASTICA$", "cirugia_plastica"),
    (r"CIRUGIA GEN|(?:CX|QX|CXS) GENERAL|^CIRUGIA$", "cirugia_general"),
    (r"MEDICINA INT|MEDI.*INTERNA|MED\.? INTERNA|M\.? INTERNA|MD INTERNA|^INTERNA$", "medicina_interna"),
    (r"MEDICINA FAM|MED\.? FAMILIAR|MEFA", "medicina_familiar"),
    (r"MEDICINA (?:DE )?(?:EMERGENCIA|URGENCIA)|EMERGENCIOLOG|^EMERGENCIAS$", "medicina_emergencias"),
    (r"MEDICINA GENERAL", "medicina_general"),
    (r"PEDIATR|PEDITR|PEDIATIR|^PED$", "pediatria"),
    (r"PSIQUI|PSQUI|PISQUI|^PSIQ$", "psiquiatria"),
    (r"ORTOP|TRAUMATOLOG", "ortopedia"),
    (r"UROLOG|^URO$", "urologia"),
    (r"NEUROLOG|^NEURO$", "neurologia"),
    (r"OFTALM|OFTAM", "oftalmologia"),
    (r"OTORR|^ORL$", "otorrinolaringologia"),
    (r"COLOPROCT", "coloproctologia"),
    (r"GINECOL|^GINECO$", "ginecologia"),
    (r"GASTRO|^GASTRO$", "gastroenterologia"),
    (r"ELECTROFISIO", "electrofisiologia"),
    (r"CARDIO", "cardiologia"),
    (r"GERIATR", "geriatria"),
    (r"TRABAJO SOCIAL|T\.? SOCIAL|^TS$", "trabajo_social"),
    (r"CLINICA (?:DE )?DOLOR|^DOLOR$", "clinica_dolor"),
    (r"REUMAT", "reumatologia"),
    (r"NEFRO", "nefrologia"),
    (r"ONCO", "oncologia"),
    (r"HEMATO", "hematologia"),
    (r"DERMATO", "dermatologia"),
    (r"PSICOLOG", "psicologia"),
    (r"RADIOLOG|RX INTERV", "radiologia_intervencionista"),
    (r"INFECTO", "infectologia"),
    (r"ENDOCRINO", "endocrinologia"),
    (r"NEUMO|PULMON", "neumologia"),
    (r"ANESTES", "anestesiologia"),
    (r"TOXICO", "toxicologia"),
    (r"PALIAT", "cuidados_paliativos"),
)

def normalize_category_text(value: object) -> str:
    """Normaliza una categoría a mayúsculas ASCII y espacios simples."""
    text = unicodedata.normalize("NFKD", str(value).upper())
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_specialty_token(token: str) -> str:
    """Asigna una etiqueta canónica mediante reglas clínicas explícitas."""
    normalized = normalize_category_text(token)
    for pattern, canonical in SPECIALTY_RULES:
        if re.search(pattern, normalized):
            return canonical
    return "otra_especialidad"


def canonicalize_specialties(value: object) -> str:
    """Separa, homologa y ordena las especialidades de un registro."""
    if pd.isna(value):
        return "especialidad_sin_registrar"
    normalized = normalize_category_text(value)
    tokens = re.split(r"\s*(?:/|\+|,|\s+-\s)\s*", str(value))
    if not normalized or not tokens:
        return "especialidad_sin_registrar"
    canonical = sorted({canonicalize_specialty_token(token) for token in tokens})
    return " | ".join(canonical)


def canonicalize_room(value: object) -> str:
    """Normaliza las salas y representa ausencias mediante categoría explícita."""
    if pd.isna(value):
        return "sala_sin_registrar"
    normalized = normalize_category_text(value)
    if normalized == "OBS CAMA 4":
        return "sala_sin_registrar"
    return normalized.lower().replace(" ", "_")


def canonicalize_triage(value: object) -> str:
    """Conserva únicamente niveles de triage del 1 al 5."""
    if pd.isna(value):
        return "triage_sin_registrar"
    match = re.fullmatch(r"TRIAGE ([1-5])", normalize_category_text(value))
    return f"triage_{match.group(1)}" if match else "triage_sin_registrar"


def canonicalize_affiliation(value: object) -> str:
    """Homologa afiliaciones y conserva INACTIVO como categoría real."""
    if pd.isna(value):
        return "afiliacion_sin_registrar"
    normalized = normalize_category_text(value)
    mapping = {
        "EJERCITO NACIONAL": "ejercito_nacional",
        "FUERZA AEREA": "fuerza_aerea",
        "ARMADA NACIONAL": "armada_nacional",
        "POLICIA NACIONAL": "policia_nacional",
        "OTRAS EAPB": "otras_eapb",
        "INACTIVO": "inactivo",
    }
    return mapping.get(normalized, "afiliacion_sin_registrar")


def canonicalize_disposition(value: object) -> str:
    """Reduce conducta a hospitalización, salida o ausencia de registro."""
    if pd.isna(value):
        return "conducta_sin_registrar"
    normalized = normalize_category_text(value)
    if normalized in {"SA", "SALIDA", "ORDEN SALIDA"}:
        return "salida"
    if (
        "HOSPITAL" in normalized
        or "REMISION" in normalized
        or normalized == "PARACLINICOS Y TRASLADO A PISO"
    ):
        return "hospitalizacion"
    return "conducta_sin_registrar"


CATEGORY_TRANSFORMATIONS = {
    "especialidades_tratantes": (
        canonicalize_specialties,
        "especialidad_sin_registrar",
    ),
    "sala_observacion": (canonicalize_room, "sala_sin_registrar"),
    "triage": (canonicalize_triage, "triage_sin_registrar"),
    "afiliacion": (canonicalize_affiliation, "afiliacion_sin_registrar"),
    "conducta": (canonicalize_disposition, "conducta_sin_registrar"),
}


def clean_selected_categories(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica todas las reglas categóricas y construye auditorías compactas."""
    cleaned = dataframe.copy(deep=True)
    audit_rows = []
    for column, (transformation, missing_label) in CATEGORY_TRANSFORMATIONS.items():
        original = cleaned[column].copy()
        cleaned[column] = original.map(transformation).astype("string")
        audit_rows.append(
            {
                "columna": column,
                "categorias_originales": int(original.nunique(dropna=True)),
                "nulos_originales": int(original.isna().sum()),
                "categorias_finales": int(cleaned[column].nunique(dropna=True)),
                "categoria_sin_registrar": missing_label,
                "registros_sin_registrar": int(cleaned[column].eq(missing_label).sum()),
            }
        )

    specialty_vocabulary = sorted(
        {
            specialty
            for value in cleaned["especialidades_tratantes"]
            for specialty in str(value).split(" | ")
        }
    )
    vocabulary = pd.DataFrame({"especialidad_homologada": specialty_vocabulary})
    specialty_row = pd.DataFrame(audit_rows)["columna"].eq(
        "especialidades_tratantes"
    )
    audit = pd.DataFrame(audit_rows)
    audit["etiquetas_individuales_finales"] = pd.NA
    audit.loc[specialty_row, "etiquetas_individuales_finales"] = len(vocabulary)
    return cleaned, audit, vocabulary
