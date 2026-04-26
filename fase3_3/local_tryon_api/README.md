# Fase 3.3 - Local TryOn API (CPU)

Proyecto base para exponer una API de virtual try-on con procesamiento local,
sin depender de proveedores cloud de pago.

## Objetivo

- Proveer endpoints HTTP para probar un flujo local de try-on.
- Mantener una arquitectura lista para reemplazar el backend base por un modelo local mas avanzado.

## Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Pillow

## Estructura

- `app/main.py`: API FastAPI
- `app/schemas.py`: modelos de request/response
- `app/service.py`: backend local base (CPU)
- `scripts/run_dev.sh`: arranque local
- `data/outputs`: resultados

## Instalacion

```bash
cd fase3_3/local_tryon_api
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar

```bash
bash scripts/run_dev.sh
```

## Endpoints

- `GET /health`: estado del servicio
- `POST /api/v1/local-tryon`: recibe `person_image` y `garment_image` (multipart) y devuelve imagen base64

## Nota

El backend actual usa un compositor local basico en CPU para validar arquitectura.
En la siguiente iteracion se puede sustituir por un modelo local mas potente.
