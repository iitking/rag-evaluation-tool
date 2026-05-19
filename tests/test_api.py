from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "RAG Evaluation" in resp.text

    resp_json = client.get("/api/info")
    assert resp_json.status_code == 200
    assert "version" in resp_json.json()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "generator_model" in data
    assert "judge_model" in data


def test_config_endpoint():
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "allowed_embedders" in data
    assert "default_strategies" in data
    assert "strategy_presets" in data
    assert "supported_generator_models" in data
    assert "supported_judge_models" in data
    assert "allowed_extensions" in data
    assert ".pdf" in data["allowed_extensions"]
    assert ".txt" in data["allowed_extensions"]
    assert ".md" in data["allowed_extensions"]


def test_sample_endpoint():
    resp = client.get("/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert "filename" in data
    assert "content" in data
    assert "questions" in data
    assert "strategies" in data
    assert "embedders" in data


def test_optimize_unsupported_file_extension():
    files = {"file": ("unsupported.csv", b"col1,col2\n1,2", "text/csv")}
    data = {"questions": "What is here?"}
    resp = client.post("/optimize", files=files, data=data)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_optimize_empty_questions():
    files = {"file": ("doc.txt", b"Some sample context text.", "text/plain")}
    data = {"questions": ""}
    resp = client.post("/optimize", files=files, data=data)
    assert resp.status_code == 400
    assert "Provide at least one question" in resp.json()["detail"]


def test_optimize_invalid_top_k():
    files = {"file": ("doc.txt", b"Some sample context text.", "text/plain")}
    data = {"questions": "What is here?", "top_k": "50"}
    resp = client.post("/optimize", files=files, data=data)
    assert resp.status_code == 400
    assert "top_k must be between 1 and 20" in resp.json()["detail"]
