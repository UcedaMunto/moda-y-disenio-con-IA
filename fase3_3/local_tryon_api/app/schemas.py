from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    backend: str


class TryOnResponse(BaseModel):
    status: str
    backend: str
    mime_type: str
    image_base64: str
