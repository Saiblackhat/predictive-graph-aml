"""
Comprehensive End-to-End API and SAR PDF Tests (Phase 8-11).
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.db.database import Base, engine, SessionLocal
from backend.db.models import User, Case

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # Check or create default analyst
    user = db.query(User).filter(User.username == "analyst").first()
    if not user:
        from backend.auth.security import get_password_hash
        user = User(username="analyst", password_hash=get_password_hash("analyst123"), role="ANALYST")
        db.add(user)
        db.commit()
    db.close()
    yield


def test_api_network():
    response = client.get("/api/network?max_nodes=15")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


def test_api_accounts_and_explanation():
    response = client.get("/api/accounts?limit=10")
    assert response.status_code == 200
    accs = response.json()
    assert len(accs) > 0

    acc_id = accs[0]["account_id"]
    detail_res = client.get(f"/api/accounts/{acc_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["account_id"] == acc_id

    expl_res = client.get(f"/api/accounts/{acc_id}/explanation")
    assert expl_res.status_code == 200
    expl = expl_res.json()
    assert "top_reasons" in expl
    assert len(expl["top_reasons"]) > 0


def test_case_management_and_sar_pdf():
    # 1. Create a case
    create_res = client.post("/api/cases", json={
        "account_id": "ACC_TEST_SAR",
        "title": "Investigation into circular transfers",
        "priority": "HIGH"
    })
    assert create_res.status_code == 200
    case_id = create_res.json()["id"]

    # 2. Add an investigation note
    note_res = client.post(f"/api/cases/{case_id}/notes", json={
        "note_text": "Traced round-trip transactions to accounts in Mumbai branch."
    })
    assert note_res.status_code == 200

    # 3. Fetch notes
    get_notes = client.get(f"/api/cases/{case_id}/notes")
    assert get_notes.status_code == 200
    assert len(get_notes.json()) >= 1

    # 4. Generate SAR PDF
    sar_res = client.post(f"/api/cases/{case_id}/sar")
    assert sar_res.status_code == 200
    assert sar_res.headers["content-type"] == "application/pdf"
    assert len(sar_res.content) > 1000  # ReportLab PDF content generated


def test_audit_trail_and_feedback():
    fb_res = client.post("/api/feedback", json={
        "account_id": "ACC_TEST_SAR",
        "label": "CONFIRMED_SUSPICIOUS",
        "comments": "Confirmed layering via smurfing"
    })
    assert fb_res.status_code == 200

    audit_res = client.get("/api/audit-trail")
    assert audit_res.status_code == 200
    logs = audit_res.json()
    assert len(logs) > 0


def test_metrics_and_settings():
    metrics = client.get("/api/metrics").json()
    assert "model_comparison" in metrics
    assert "stats" in metrics

    settings = client.get("/api/settings").json()
    assert "structuring_limit" in settings
