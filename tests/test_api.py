"""
Pruebas Unitarias y de Integracion para la API REST de Retail Intelligence Platform.
Prueba los endpoints /health, /predict y /benchmark, asi como el manejo de errores.
"""

import pytest
from fastapi.testclient import TestClient
from retail_platform.api.app import app


@pytest.fixture(scope="module")
def client():
    """Fixture que inicializa el cliente de prueba ejecutando el lifespan (carga de modelo y datos)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifica que /health responda 200 y reporte el modelo XGBoost cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "XGBoost" in data["model_name"]
    assert data["features_loaded"] is True
    assert data["total_series_available"] == 54 * 33  # 1782 series


def test_benchmark_endpoint(client):
    """Verifica que /benchmark retorne la tabla comparativa con el modelo campeon."""
    response = client.get("/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "validation_period" in data
    assert len(data["results"]) >= 4
    assert data["champion_model"] == "XGBoost Global Multi-Serie"

    # Verificar que XGBoost este presente
    models = [r["model"] for r in data["results"]]
    assert any("XGBoost" in m for m in models)


def test_predict_endpoint_success(client):
    """Verifica una prediccion valida para Tienda 1 y GROCERY I en el periodo de test."""
    payload = {
        "store_nbr": 1,
        "family": "GROCERY I",
        "start_date": "2017-08-16",
        "end_date": "2017-08-31"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["store_nbr"] == 1
    assert data["family"] == "GROCERY I"
    assert data["forecast_horizon"] == 16
    assert len(data["forecasts"]) == 16
    for f in data["forecasts"]:
        assert "date" in f
        assert "predicted_sales" in f
        assert f["predicted_sales"] >= 0.0


def test_predict_invalid_family(client):
    """Verifica que una familia inexistente retorne error 404."""
    payload = {
        "store_nbr": 1,
        "family": "INVALID_FAMILY_XYZ",
        "start_date": "2017-08-16",
        "end_date": "2017-08-31"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    assert "no encontrada" in response.json()["detail"]


def test_predict_invalid_store(client):
    """Verifica que un numero de tienda fuera de rango retorne error de validacion 422."""
    payload = {
        "store_nbr": 999,  # Solo hay tiendas 1 a 54
        "family": "GROCERY I",
        "start_date": "2017-08-16",
        "end_date": "2017-08-31"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error (le=54)
