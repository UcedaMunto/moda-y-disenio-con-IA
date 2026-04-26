from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import HealthResponse, TryOnResponse
from app.service import LocalTryOnService


app = FastAPI(
    title="Local TryOn API",
    version="0.1.0",
    description="API local para virtual try-on sin proveedores cloud de pago.",
)

service = LocalTryOnService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", backend="local_cpu_baseline")


@app.post("/api/v1/local-tryon", response_model=TryOnResponse)
async def local_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
) -> TryOnResponse:
    if not person_image.content_type or not person_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="person_image debe ser una imagen")
    if not garment_image.content_type or not garment_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="garment_image debe ser una imagen")

    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    if not person_bytes or not garment_bytes:
        raise HTTPException(status_code=400, detail="Archivos vacios no permitidos")

    result = service.run(person_bytes, garment_bytes)

    output_dir = Path(os.getenv("LOCAL_TRYON_OUTPUT_DIR", "data/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return TryOnResponse(
        status="ok",
        backend="local_cpu_baseline",
        mime_type=result.mime_type,
        image_base64=result.image_base64,
    )
