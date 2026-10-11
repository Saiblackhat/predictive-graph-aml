"""
Unit tests for Endpoints, Cases, SAR generation, and Authentication (Sprints 2, 3, 4, 6).
"""

from fastapi.testclient import TestClient
from server.main import app
from src.reporting.sar_generator import generate_sar_pdf

client = TestClient(app)


def test_auth_login():
    resp = client.post("/api/auth/login", json={"username": "analyst", "password": "AnalystPass123!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == "ANALYST"


def test_accounts_search():
    resp = client.get("/api/accounts/search?query=ACC")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_cases_workflow():
    # 1. Create case
    c_resp = client.post("/api/cases", json={"account_id": "ACC_00001", "title": "Test review case"})
    assert c_resp.status_code == 200
    case_id = c_resp.json()["id"]

    # 2. Add note
    n_resp = client.post(f"/api/cases/{case_id}/notes", json={"note_text": "Investigated node clustering"})
    assert n_resp.status_code == 200

    # 3. Update status
    u_resp = client.patch(f"/api/cases/{case_id}", json={"status": "IN_REVIEW"})
    assert u_resp.status_code == 200
    assert u_resp.json()["status"] == "IN_REVIEW"

    # 4. Generate SAR PDF
    sar_resp = client.post(f"/api/cases/{case_id}/sar")
    assert sar_resp.status_code == 200
    assert sar_resp.headers["content-type"] == "application/pdf"
    assert len(sar_resp.content) > 1000


def test_sar_pdf_generator_standalone():
    sample_data = {
        "case_id": 999,
        "account_id": "ACC_LAUND_TEST",
        "risk_score": 92.5,
        "risk_tier": "CRITICAL",
        "typology": "CIRCULAR",
        "status": "OPEN",
        "created_at": "2026-10-11",
        "shap_reasons": [{"plain_language": "Part of a 4-account loop", "impact": "RAISES_RISK"}],
        "transactions": [{"timestamp": "2026-01-01", "direction": "OUT", "counterparty": "ACC_02", "amount": 25000.0}],
        "notes": [{"author": "analyst", "text": "Rapid pass-through validated", "date": "2026-10-11"}],
    }
    pdf_bytes = generate_sar_pdf(sample_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
