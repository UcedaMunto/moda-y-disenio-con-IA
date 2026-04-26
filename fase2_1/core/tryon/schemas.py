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
    pose_backend: str = Field(
        default="landmarker",
        pattern="^(legacy|landmarker|both)$",
        description="Backend de pose: legacy, landmarker o both (fallback)",
    )
    apply_pose_guides: bool = Field(
        default=False,
        description="Si aplica guias de pose para ajustar escala/angulo de la prenda",
    )
    pose_guide_strength: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Fuerza de mezcla de guias de pose sobre transformacion base",
    )
    size_multiplier: float = Field(
        default=1.0,
        ge=0.6,
        le=1.8,
        description="Escala manual adicional para aumentar/reducir tamanio final",
    )
    garment_in_front: bool = Field(
        default=False,
        description="Si true, dibuja prenda al frente y desactiva oclusion por mascara",
    )
    shape_guide_keypoints: dict[str, dict] | None = Field(
        default=None,
        description="Keypoints de Human Shape Lab para guiar escala/posicion de superposicion",
    )
    detect_and_replace_garment: bool = Field(
        default=False,
        description="Si true, detecta la prenda actual en la imagen y la reemplaza con la textura",
    )
    remove_clothing_and_apply_texture: bool = Field(
        default=False,
        description="NUEVO: Si true, detecta TODA la ropa, la hace transparente y aplica textura de tela seleccionada",
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
