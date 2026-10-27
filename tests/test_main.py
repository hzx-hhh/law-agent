import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


client = TestClient(main.app)


def test_health_and_readiness():
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").status_code == 200


def test_validation_rejects_short_input():
    response = client.post("/api/analyze", json={"case_description": "太短", "claims": ["押金"]})
    assert response.status_code == 422


def test_degraded_mode_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    response = client.post("/api/analyze", headers={"X-Request-ID": "req_test_001"}, json={
        "case_description": "承租人签订一年租赁合同，居住四个月后因工作调动提前退租。房东以墙面污损为由拒绝返还押金，未提供维修发票。",
        "claims": ["返还押金", "减少扣款"],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "DEGRADED"
    assert data["request_id"] == "req_test_001"
    assert data["citations"]
    assert "律师意见" in data["disclaimer"]
    assert response.headers["X-Request-ID"] == "req_test_001"


def test_retrieval_is_explainable():
    db = main.load_laws_db()
    result = main.retrieve_relevant_context("提前退租，家具损坏，要求退还押金", ["返还押金"], db)
    assert result["laws"]
    assert result["cases"]
    assert "scores" in result


def test_pii_is_redacted_before_model_context():
    text = "张三身份证号110101199001011234，手机号13800138000"
    redacted = main.redact_pii(text)
    assert "110101199001011234" not in redacted
    assert "13800138000" not in redacted
    assert "身份证" in redacted and "手机号" in redacted


def test_eval_data_is_valid_jsonl():
    path = Path(__file__).parents[1] / "eval_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 4
    assert all("case_id" in row and "expected" in row for row in rows)


def test_token_upload_and_metrics():
    token = client.post("/api/auth/token", json={"tenant_id": "tenant-a", "user_id": "user-a"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    upload = client.post("/api/documents/upload", headers=headers, files={"file": ("notes.txt", "身份证号110101199001011234，押金争议", "text/plain")})
    assert upload.status_code == 200
    assert "110101199001011234" not in upload.json()["extracted_text"]
    assert client.get("/metrics").status_code == 200


def test_idempotency_returns_same_response():
    headers = {"Idempotency-Key": "same-request-001"}
    body = {"case_description": "承租人提前退租，房东以墙面污损为由拒绝返还押金，未提供维修发票。", "claims": ["返还押金"]}
    first = client.post("/api/analyze", headers=headers, json=body)
    second = client.post("/api/analyze", headers=headers, json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["request_id"] == second.json()["request_id"]


def test_docx_export_after_analysis():
    body = {"case_description": "承租人提前退租，房东以墙面污损为由拒绝返还押金，未提供维修发票。", "claims": ["返还押金"]}
    result = client.post("/api/analyze", json=body).json()
    response = client.get(f"/api/analyses/{result['analysis_id']}/export.docx")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert len(response.content) > 1000


def test_corpus_metadata_and_hybrid_retrieval():
    db = main.load_laws_db()
    assert db["metadata"]["source_url"]
    assert all(item.get("effective_from") and item.get("source_url") for item in db["laws"])
    from retrieval import hybrid_search
    results = hybrid_search("押金退还", db["laws"])
    assert results and results[0].document["id"]


def test_retrieval_debug_endpoint():
    response = client.get("/api/retrieval/search?q=押金退还")
    assert response.status_code == 200
    assert response.json()["results"]
    assert response.json()["backend"] in {"lexical-fallback", "embedding+lexical"}


def test_invalid_model_output_is_not_reported_as_success(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("模型输出缺少 issues 或 draft 字段")
    monkeypatch.setattr(main, "call_llm_analysis", broken)
    response = client.post("/api/analyze", json={"case_description": "承租人签订一年租赁合同，实际居住四个月后因工作调动提前退租，出租人拒绝返还押金并主张家具损坏，但没有提供维修票据。", "claims": ["返还押金"]})
    assert response.status_code == 200
    assert response.json()["status"] == "MODEL_OUTPUT_INVALID"


def test_upload_rejects_unsupported_type_and_oversized_file(monkeypatch):
    unsupported = client.post("/api/documents/upload", files={"file": ("malware.exe", b"MZ", "application/octet-stream")})
    assert unsupported.status_code == 415
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 2)
    oversized = client.post("/api/documents/upload", files={"file": ("large.txt", b"abcd", "text/plain")})
    assert oversized.status_code == 413


def test_failure_case_catalog_is_complete():
    path = Path(__file__).parents[1] / "failure_cases.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    assert all(row["trigger"] and row["expected_status"] and row["regression"] for row in rows)


def test_real_bm25_changes_score_from_plain_overlap():
    from retrieval import hybrid_search
    documents = [
        {"id": "a", "title": "押金退还", "content": "押金 退还 押金 退还"},
        {"id": "b", "title": "房屋维修", "content": "房屋维修和正常损耗"},
    ]
    results = hybrid_search("押金退还", documents)
    assert results[0].document["id"] == "a"
    assert results[0].bm25_score > 0


def test_auth_can_be_required(monkeypatch):
    monkeypatch.setenv("REQUIRE_AUTH", "true")
    response = client.get("/api/cases")
    assert response.status_code == 401
    monkeypatch.setenv("REQUIRE_AUTH", "false")


def test_cross_tenant_cannot_export_analysis():
    token_a = client.post("/api/auth/token", json={"tenant_id": "tenant-a-isolation", "user_id": "user-a"}).json()["access_token"]
    token_b = client.post("/api/auth/token", json={"tenant_id": "tenant-b-isolation", "user_id": "user-b"}).json()["access_token"]
    body = {"case_description": "承租人签订一年租赁合同，居住四个月后因工作调动提前退租，出租人拒绝返还押金并主张家具损坏。", "claims": ["返还押金"]}
    result = client.post("/api/analyze", headers={"Authorization": f"Bearer {token_a}"}, json=body).json()
    response = client.get(f"/api/analyses/{result['analysis_id']}/export", headers={"Authorization": f"Bearer {token_b}"})
    assert response.status_code == 404
