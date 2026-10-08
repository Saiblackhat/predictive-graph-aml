"""Controlled Multilingual Clinical Translation for Phase 6.

Translates verified English patient instructions into Spanish ('es')
using deterministic, meaning-preserving templates.

CORE INVARIANTS:
1. Never translate medication names or formulation salts (e.g. 'metoprolol tartrate').
2. Numbers, units, and decimal quantities are strictly preserved.
3. Inequalities ('more than 3 pounds' -> 'más de 3 libras') are strictly preserved.
4. Negation ('Do not take' -> 'No tome') is strictly preserved.
5. Timeframe operators ('within 7 days' -> 'dentro de 7 días', never 'después de') are preserved.
6. Safety status is strictly inherited without upgrade.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.schemas.models import MedicationAction
from backend.task_orchestration.types import TaskCategory
from backend.patient_language.types import (
    LanguageCode,
    PatientLanguageItem,
    TransformationStatus,
    TranslatedPatientLanguageItem,
)

SPANISH_FREQUENCY_MAP: dict[str, str] = {
    "twice a day": "dos veces al día",
    "three times a day": "tres veces al día",
    "four times a day": "cuatro veces al día",
    "once a day": "una vez al día",
    "every morning": "todas las mañanas",
    "every evening": "todas las tardes",
    "every night": "todas las noches",
    "at bedtime": "al acostarse",
    "as needed": "según sea necesario",
}

SPANISH_ROUTE_MAP: dict[str, str] = {
    "by mouth": "por vía oral",
    "orally": "por vía oral",
    "po": "por vía oral",
    "through an iv": "por vía intravenosa",
    "through an IV": "por vía intravenosa",
    "iv": "por vía intravenosa",
    "intravenously": "por vía intravenosa",
    "as an injection under the skin": "por inyección subcutánea",
    "subcutaneously": "por inyección subcutánea",
    "subq": "por inyección subcutánea",
    "under your tongue": "debajo de la lengua",
    "sublingually": "debajo de la lengua",
    "sublingual": "debajo de la lengua",
    "on your skin": "en la piel",
    "topically": "en la piel",
    "in your nose": "en la nariz",
    "nasally": "en la nariz",
}

SPANISH_TIMEFRAME_MAP: dict[str, str] = {
    "within 7 days": "dentro de 7 días",
    "within 14 days": "dentro de 14 días",
    "within 30 days": "dentro de 30 días",
    "after 7 days": "después de 7 días",
    "after 14 days": "después de 14 días",
    "in 5 days": "en 5 días",
    "in 7 days": "en 7 días",
    "in 14 days": "en 14 días",
    "in 30 days": "en 30 días",
}


def translate_timeframe_to_spanish(tf_display: Optional[str]) -> Optional[str]:
    """Translate relative or absolute timeframe into Spanish without corrupting operators."""
    if not tf_display:
        return None
    lower = tf_display.strip().lower()
    if lower in SPANISH_TIMEFRAME_MAP:
        return SPANISH_TIMEFRAME_MAP[lower]

    # Regex fallback for "within X days"
    m_within = re.match(r"^within\s+(\d+)\s*(days?|weeks?|months?)$", lower)
    if m_within:
        n, u = m_within.groups()
        u_es = "días" if "day" in u else ("semanas" if "week" in u else "meses")
        return f"dentro de {n} {u_es}"

    m_after = re.match(r"^after\s+(\d+)\s*(days?|weeks?|months?)$", lower)
    if m_after:
        n, u = m_after.groups()
        u_es = "días" if "day" in u else ("semanas" if "week" in u else "meses")
        return f"después de {n} {u_es}"

    m_in = re.match(r"^in\s+(\d+)\s*(days?|weeks?|months?)$", lower)
    if m_in:
        n, u = m_in.groups()
        u_es = "días" if "day" in u else ("semanas" if "week" in u else "meses")
        return f"en {n} {u_es}"

    return tf_display


def translate_patient_item_to_spanish(
    item: PatientLanguageItem,
    trans_index: int,
) -> TranslatedPatientLanguageItem:
    """Deterministically translate a PatientLanguageItem to Spanish."""
    meta = item.preserved_fields
    cat = item.category

    translated_title: str
    translated_instruction: str

    if cat == TaskCategory.MEDICATION:
        med_name = meta.get("medication_name", "su medicamento")
        dose = meta.get("dosage")
        dose_str = f" {dose}" if dose else ""
        action = meta.get("action", MedicationAction.UNKNOWN.value)

        freq_en = meta.get("frequency_expanded", "")
        freq_es = SPANISH_FREQUENCY_MAP.get(freq_en.lower(), freq_en) if freq_en else ""
        freq_str = f" {freq_es}" if freq_es else ""

        route_en = meta.get("route", "")
        route_exp = meta.get("route_expanded", "")
        route_es = (
            SPANISH_ROUTE_MAP.get(route_exp.lower())
            or SPANISH_ROUTE_MAP.get(route_en.lower())
            or route_exp
            or route_en
        ) if (route_en or route_exp) else ""
        route_str = f" {route_es}" if route_es else ""

        if action == MedicationAction.STOP_DISCONTINUE.value:
            translated_title = f"Deje de tomar {med_name}{dose_str}"
            translated_instruction = f"No tome {med_name}{dose_str}. Deje de tomar este medicamento después del alta hospitalaria."
        elif action == MedicationAction.HOLD_TEMPORARY.value:
            translated_title = f"Pausar temporalmente {med_name}{dose_str}"
            translated_instruction = f"No tome {med_name}{dose_str} en este momento. Pausa temporal según las indicaciones médicas."
        elif action == MedicationAction.NEW_START.value:
            translated_title = f"Comience a tomar {med_name}{dose_str}"
            translated_instruction = f"Comience a tomar su nuevo medicamento {med_name}{dose_str}{freq_str}{route_str}."
        elif action == MedicationAction.DOSE_CHANGE.value:
            prev_dose = meta.get("previous_dosage")
            prev_str = f" de {prev_dose}" if prev_dose else ""
            translated_title = f"Cambie su dosis de {med_name} a{dose_str}"
            translated_instruction = f"Cambie su dosis de {med_name}{prev_str} a{dose_str}{freq_str}{route_str}."
        elif action == MedicationAction.CONTINUE.value:
            translated_title = f"Continúe tomando {med_name}{dose_str}"
            translated_instruction = f"Continúe tomando {med_name}{dose_str}{freq_str}{route_str}."
        else:
            translated_title = f"Revise la receta de {med_name}"
            translated_instruction = f"Hable con su médico antes de tomar {med_name}{dose_str}. Requiere verificación clínica."

    elif cat == TaskCategory.FOLLOW_UP:
        spec = meta.get("specialty", "Especialista")
        prov = meta.get("provider_name")
        prov_str = f" con {prov}" if prov else ""
        tf_es = translate_timeframe_to_spanish(meta.get("timeframe"))
        tf_str = f" {tf_es}" if tf_es else ""

        translated_title = f"Cita de seguimiento con {spec}{prov_str}"
        translated_instruction = f"Programe y asista a una cita con {spec}{prov_str}{tf_str}."

    elif cat == TaskCategory.TEST:
        test_name = meta.get("test_name", "Prueba médica")
        tf_es = translate_timeframe_to_spanish(meta.get("timeframe"))
        tf_str = f" {tf_es}" if tf_es else ""

        translated_title = f"Prueba de laboratorio: {test_name}"
        translated_instruction = f"Realice esta prueba de laboratorio: {test_name}{tf_str}."

    elif cat == TaskCategory.REFERRAL:
        spec = meta.get("specialty_or_service", "Servicio especializado")
        prov = meta.get("provider_name")
        prov_str = f" con {prov}" if prov else ""
        tf_es = translate_timeframe_to_spanish(meta.get("timeframe"))
        tf_str = f" {tf_es}" if tf_es else ""

        translated_title = f"Remisión: {spec}{prov_str}"
        translated_instruction = f"Programe una cita para {spec}{prov_str}{tf_str}."

    elif cat == TaskCategory.WARNING_MONITORING:
        symp = meta.get("symptom_description", "signos de alarma")
        # Translate inequality phrases while preserving numbers & units exactly
        symp_es = symp
        symp_es = re.sub(r"\bweight increases by more than (\d+(?:\.\d+)?)\s*pounds in 24 hours\b", r"el peso aumenta en más de \1 libras en 24 horas", symp_es, flags=re.IGNORECASE)
        symp_es = re.sub(r"\btemperature is greater than (\d+(?:\.\d+)?)\s*F\b", r"la temperatura es mayor de \1 F", symp_es, flags=re.IGNORECASE)
        symp_es = re.sub(r"\bchest pain\b", "dolor en el pecho", symp_es, flags=re.IGNORECASE)

        action = meta.get("action_to_take", "Llame a la clínica")
        translated_title = f"Alerta de signo de alarma: {symp_es}"
        translated_instruction = f"Llame a la clínica de inmediato si {symp_es}. Acción: {action}."

    else:
        translated_title = item.simplified_title
        translated_instruction = item.simplified_instruction

    return TranslatedPatientLanguageItem(
        item_id=f"trans-{trans_index:03d}",
        source_language_item_id=item.item_id,
        source_task_id=item.source_task_id,
        source_language=LanguageCode.EN,
        target_language=LanguageCode.ES,
        translated_title=translated_title,
        translated_instruction=translated_instruction,
        transformation_status=item.transformation_status,
        patient_releasable=item.patient_releasable,
        requires_human_review=item.requires_human_review,
        review_reasons=list(item.review_reasons),
        evidence=item.evidence,
        original_source_text=item.original_source_text,
        preserved_fields=item.preserved_fields,
    )
