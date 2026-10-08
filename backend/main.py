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
