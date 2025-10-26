# tests/test_forecast_api.py
from fastapi.testclient import TestClient
from app.main import create_app

client = TestClient(create_app())

def test_forecast_basic():
    resp = client.get("/api/v1/forecast/AAPL?horizons=7,30&method=naive-drift")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # 顶层字段
    assert data["ticker"] == "AAPL"
    assert "generated_at" in data
    assert "current_price" in data
    assert data["method"] in ("naive-drift", "ma")  # 取决于传参

    # 预测列表
    preds = data["predictions"]
    assert isinstance(preds, list) and len(preds) == 2
    horizons = sorted(p["horizon_days"] for p in preds)
    assert horizons == [7, 30]

    for p in preds:
        assert "predicted" in p
        assert p["predicted"] > 0
        # 置信度可选，若存在则在 [0,1]
        if "confidence" in p and p["confidence"] is not None:
            assert 0.0 <= p["confidence"] <= 1.0

def test_forecast_method_switch():
    for method in ("naive-drift", "ma"):
        resp = client.get(f"/api/v1/forecast/MSFT?horizons=7&method={method}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["method"] == method
