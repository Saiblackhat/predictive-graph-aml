"""FastAPI application entrypoint for CAREBRIDGE foundation."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.fixtures.discharge_cases import ALL_SYNTHETIC_CASES
from backend.safety.validator import validate_discharge_payload
from backend.schemas.models import (
    MANDATORY_CLINICAL_DISCLAIMER,
    MAX_RAW_DISCHARGE_TEXT_BYTES,
    DischargeExtractionPayload,
    RawTextTooLargeError,
    ValidationReport,
)

app = FastAPI(
    title="CAREBRIDGE API",
    description="Agentic Hospital Discharge & Follow-up Coordinator (Clinical Safety First)",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return HTTP 413 when raw_discharge_text exceeds 200 KB; otherwise HTTP 422."""
    for err in exc.errors():
        loc = err.get("loc", ())
        msg = str(err.get("msg", ""))
        err_type = str(err.get("type", ""))
        ctx_err = err.get("ctx", {}).get("error")
        if isinstance(ctx_err, RawTextTooLargeError) or (
            "raw_discharge_text" in loc
            and (
                err_type == "string_too_long"
                or "exceeds maximum size" in msg
                or str(MAX_RAW_DISCHARGE_TEXT_BYTES) in msg
            )
        ):
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"raw_discharge_text exceeds maximum allowed size of {MAX_RAW_DISCHARGE_TEXT_BYTES} bytes (200 KB)."
                },
            )
    return JSONResponse(status_code=422, content={"detail": str(exc.errors())})


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health and mandatory clinical safety boundary statement."""
    return {
        "status": "ok",
        "service": "CAREBRIDGE",
        "disclaimer": MANDATORY_CLINICAL_DISCLAIMER,
    }


@app.post("/api/v1/validate", response_model=ValidationReport)
def validate_payload_endpoint(payload: DischargeExtractionPayload) -> ValidationReport:
    """Validate a structured discharge extraction payload through the Safety Validation Layer."""
    return validate_discharge_payload(payload)


@app.get("/api/v1/fixtures/{fixture_name}", response_model=ValidationReport)
def evaluate_synthetic_fixture(fixture_name: str) -> ValidationReport:
    """Load and validate one of the five canonical synthetic discharge fixtures."""
    builder = ALL_SYNTHETIC_CASES.get(fixture_name.lower())
    if builder is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown fixture '{fixture_name}'. Valid options: {list(ALL_SYNTHETIC_CASES.keys())}",
        )
    payload = builder()
    return validate_discharge_payload(payload)


# ─── Phase 8 — Unified Application Endpoints ─────────────────────────────────

from pathlib import Path
from typing import Any
from fastapi.responses import HTMLResponse
from backend.api_orchestrator import (
    get_available_cases,
    orchestrate_pipeline,
    search_providers_endpoint,
    update_notification_state,
    update_patient_language,
)


@app.get("/api/v1/cases")
def list_cases_endpoint() -> list[dict[str, Any]]:
    """Return available synthetic cases for the CareBridge command center."""
    return get_available_cases()


@app.post("/api/v1/orchestrate")
def orchestrate_endpoint(req: dict[str, Any]) -> dict[str, Any]:
    """Execute end-to-end clinical intelligence pipeline and return full UI state."""
    case_id = req.get("case_id")
    raw_text = req.get("raw_text")
    language = req.get("language")
    return orchestrate_pipeline(case_id=case_id, raw_text=raw_text, language=language)


@app.patch("/api/v1/patients/{patient_id}/language")
def update_patient_language_endpoint(patient_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Update patient preferred language and re-translate scheduled notifications."""
    preferred_language = req.get("preferred_language") or req.get("language")
    if not preferred_language or not isinstance(preferred_language, str):
        raise HTTPException(status_code=422, detail="Field 'preferred_language' is required.")
    return update_patient_language(patient_id=patient_id, new_language=preferred_language)


@app.post("/api/v1/providers/search")
def search_providers_api(req: dict[str, Any]) -> dict[str, Any]:
    """Attribute-based provider search without clinical recommendation or ranking."""
    specialty = req.get("specialty", "Cardiology")
    location_city = req.get("location_city")
    language = req.get("language")
    insurance = req.get("insurance")
    return search_providers_endpoint(
        specialty=specialty,
        location_city=location_city,
        language=language,
        insurance=insurance,
    )


@app.post("/api/v1/notifications/{notif_id}/action")
def notification_action_endpoint(notif_id: str, req: dict[str, Any]) -> dict[str, Any]:
    """Transition notification state (ACKNOWLEDGE, DISMISS, CANCEL, SNOOZE)."""
    action = req.get("action", "ACKNOWLEDGE")
    snooze_minutes = req.get("snooze_minutes", 60)
    return update_notification_state(notif_id, action, snooze_minutes=snooze_minutes)


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def serve_app() -> HTMLResponse:
    """Serve CareBridge Cinematic Command Center application."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    # Fallback to frontend index if present
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>CareBridge API Active</h1><p>Cinematic UI building...</p>")
