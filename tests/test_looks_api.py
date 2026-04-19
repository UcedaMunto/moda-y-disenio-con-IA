"""Tests para endpoints de looks y prueba virtual."""
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from apps.api.main import app, LOOKS_DIR

client = TestClient(app)


def test_list_looks_empty_or_present():
    resp = client.get("/looks")
    assert resp.status_code == 200
    data = resp.json()
    assert "looks" in data
    assert "total" in data
    assert isinstance(data["looks"], list)


def test_save_look_returns_look_id(tmp_path, monkeypatch):
    monkeypatch.setattr("apps.api.main.LOOKS_DIR", tmp_path)
    resp = client.post("/looks/save", json={
        "name": "Camiseta test",
        "model_path": "data/raw/models/TShirts.obj",
        "texture_path": "data/raw/telas/gris.webp",
        "garment_type": "shirt",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    assert "look_id" in data
    assert data["look"]["name"] == "Camiseta test"


def test_save_look_missing_name():
    resp = client.post("/looks/save", json={
        "name": "",
        "model_path": "data/raw/models/TShirts.obj",
    })
    assert resp.status_code == 422


def test_list_fotos_personas():
    resp = client.get("/fotos-personas")
    assert resp.status_code == 200
    data = resp.json()
    assert "fotos" in data
    assert "total" in data
    # fotos_personas directory has 6 images
    assert data["total"] >= 1


def test_tryon_apply_look_not_found():
    resp = client.post("/tryon/apply", json={
        "look_id": "nonexistent_look",
        "foto_nombre": "test.jpg",
    })
    assert resp.status_code == 404


def test_tryon_page_returns_html():
    resp = client.get("/tryon")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
