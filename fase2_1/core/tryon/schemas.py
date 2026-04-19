from pydantic import BaseModel, Field


class TryOnRequest(BaseModel):
    image_path: str = Field(..., description="Ruta de la imagen de persona")
    garment_path: str = Field(..., description="Ruta del recurso de prenda")
    output_path: str = Field(..., description="Ruta de salida para imagen final")
    garment_type: str = Field(
        default="other",
        pattern="^(shirt|skirt|pants|dress|other)$",
        description="Categoria de prenda para ajuste por tipo",
    )


class TryOnResult(BaseModel):
    status: str
    output_path: str
    scale: float | None = None
    meta: dict | None = None


class TryOnBatchRequest(BaseModel):
    input_dir: str = Field(..., description="Directorio de imagenes de persona")
    garment_path: str = Field(..., description="Ruta de prenda usada en batch")
    garment_type: str = Field(
        default="other",
        pattern="^(shirt|skirt|pants|dress|other)$",
        description="Categoria de prenda usada en batch",
    )
    output_dir: str = Field(default="data/processed/fase2_1_eval/outputs", description="Directorio de salidas")
    report_path: str = Field(default="data/processed/fase2_1_eval/report.json", description="Ruta de reporte JSON")
    checklist_path: str | None = Field(default=None, description="Ruta opcional del checklist de calidad")
    limit: int | None = Field(default=None, ge=1, le=10000)


class TryOnBatchResult(BaseModel):
    status: str
    summary: dict
    report_path: str
    results: list[dict]


class TryOnManualEvalRequest(BaseModel):
    image: str = Field(..., description="Ruta de imagen evaluada")
    output_path: str | None = Field(default=None, description="Ruta de salida try-on asociada")
    criteria_scores: dict[str, float] = Field(default_factory=dict, description="Mapa criterio->puntaje [0,1]")
    reviewer: str | None = Field(default=None, description="Nombre/identificador del revisor")
    comment: str | None = Field(default=None, description="Comentario libre")
    report_path: str = Field(
        default="data/processed/fase2_1_eval/manual_reviews.jsonl",
        description="Ruta de persistencia JSONL",
    )
    checklist_path: str | None = Field(default=None, description="Ruta opcional del checklist de calidad")
    project_id: str = Field(default="fase2_1", description="Identificador del proyecto")


class TryOnManualEvalResult(BaseModel):
    status: str
    saved_path: str
    score: float
    record: dict
