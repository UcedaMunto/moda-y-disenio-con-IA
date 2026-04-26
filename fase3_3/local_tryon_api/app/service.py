from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class LocalTryOnResult:
    mime_type: str
    image_base64: str


class LocalTryOnService:
    """Backend local base para try-on en CPU.

    Implementa una composicion simple para validar la API local.
    """

    def __init__(self) -> None:
        alpha_raw = os.getenv("LOCAL_TRYON_DEFAULT_ALPHA", "0.55")
        try:
            alpha_value = float(alpha_raw)
        except ValueError:
            alpha_value = 0.55
        self.alpha = max(0.0, min(1.0, alpha_value))

    def run(self, person_image_bytes: bytes, garment_image_bytes: bytes) -> LocalTryOnResult:
        person = Image.open(io.BytesIO(person_image_bytes)).convert("RGBA")
        garment = Image.open(io.BytesIO(garment_image_bytes)).convert("RGBA")

        # Ajusta prenda al tamano de la persona con un margen para superposicion centrada.
        width, height = person.size
        garment = garment.resize((width, height), Image.Resampling.LANCZOS)

        blended = Image.blend(person, garment, self.alpha).convert("RGB")

        output = io.BytesIO()
        blended.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")

        return LocalTryOnResult(
            mime_type="image/png",
            image_base64=encoded,
        )
