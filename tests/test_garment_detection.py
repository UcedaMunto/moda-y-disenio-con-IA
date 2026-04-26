"""Tests para detección y reemplazo de prendas."""
from pathlib import Path

import numpy as np
from PIL import Image

from fase2_1.core.tryon.schemas import TryOnRequest
from fase2_2.core.tryon.pipeline_v2 import run_tryon_v2


def _write_rgb(path: Path, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((480, 640, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_texture(path: Path, color: tuple = (255, 128, 64)) -> None:
    """Escribe textura de prueba con color específico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((256, 256, 3), color, dtype=np.uint8)
    # Agrega patrón de textura
    for i in range(0, 256, 16):
        arr[i::32, :, :] = np.clip(arr[i::32, :, :] - 30, 0, 255)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_person_with_shirt(path: Path) -> None:
    """Escribe imagen de persona con camiseta detectables (color uniforme en torso)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fondo blanco
    arr = np.full((480, 640, 3), 220, dtype=np.uint8)
    
    # Piel (cara/cuello/brazos)
    skin_color = np.array([210, 170, 140], dtype=np.uint8)
    arr[50:150, 250:390] = skin_color  # Cara
    arr[150:180, 200:340] = skin_color  # Cuello
    arr[150:300, 150:250] = skin_color  # Brazo izqierdo
    arr[150:300, 390:490] = skin_color  # Brazo derecho
    
    # Camiseta (color naranja uniforme en torso)
    shirt_color = np.array([255, 140, 30], dtype=np.uint8)
    arr[180:350, 220:420] = shirt_color  # Torso
    
    # Pantalón (gris)
    pants_color = np.array([100, 100, 100], dtype=np.uint8)
    arr[350:450, 240:400] = pants_color  # Pantalón
    
    # Piernas (piel)
    arr[450:480, 260:380] = skin_color
    
    Image.fromarray(arr, mode="RGB").save(path)


def _write_body_mask(path: Path, img: np.ndarray) -> None:
    """Escribe máscara de cuerpo basada en imagen (simplificado para test)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Máscara simple: todo lo que no es fondo blanco (>200 en todos los canales)
    mask = np.where(
        (img[:, :, 0] < 200) | (img[:, :, 1] < 200) | (img[:, :, 2] < 200),
        255,
        0
    ).astype(np.uint8)
    
    Image.fromarray(mask, mode="L").save(path)


def test_garment_detection_and_replacement(tmp_path: Path, monkeypatch) -> None:
    """Test de detección y reemplazo de prenda."""
    person = tmp_path / "person.jpg"
    texture = tmp_path / "texture.png"
    output = tmp_path / "output.jpg"
    mask = tmp_path / "mask.png"
    
    # Crea imagen de persona con camiseta
    _write_person_with_shirt(person)
    person_arr = np.array(Image.open(person))
    _write_body_mask(mask, person_arr)
    
    # Crea textura
    _write_texture(texture, color=(200, 100, 50))
    
    # Mock de detección de pose (no es necesario para detección de prenda)
    class _Landmark:
        def __init__(self, x: float = 0.0, y: float = 0.0):
            self.x = x
            self.y = y
            self.z = 0.0
            self.visibility = 1.0
    
    class _MockLandmarks:
        def __init__(self):
            self.landmark = [_Landmark(0.0, 0.0) for _ in range(33)]
            # Configura puntos clave mínimos para que funcione compute_scale
            self.landmark[11] = _Landmark(0.3, 0.2)  # left_shoulder
            self.landmark[12] = _Landmark(0.7, 0.2)  # right_shoulder
            self.landmark[23] = _Landmark(0.35, 0.7)  # left_hip
            self.landmark[24] = _Landmark(0.65, 0.7)  # right_hip
    
    monkeypatch.setattr("fase2_2.core.tryon.pipeline_v2.detect_pose", lambda _: _MockLandmarks())
    
    # Mock de segmentación
    def _mock_segment_person_v2(image_path, output_mask_path=None, threshold=0.35, model_config_path=None):
        if output_mask_path:
            import shutil
            if Path(mask).exists():
                shutil.copy(mask, output_mask_path)
        from fase2_1.core.tryon.parsing import SegmentationResult
        return SegmentationResult(
            mask_path=str(output_mask_path) if output_mask_path else str(mask),
            backend="test",
            note="ok",
            mask_coverage=40.0
        )
    
    monkeypatch.setattr(
        "fase2_2.core.tryon.pipeline_v2.segment_person_v2",
        _mock_segment_person_v2
    )
    
    # Test 1: En modo tradicional (overlay) - debería funcionar
    req = TryOnRequest(
        image_path=str(person),
        garment_path=str(texture),
        output_path=str(tmp_path / "output_overlay.jpg"),
        garment_type="shirt",
        detect_and_replace_garment=False,
    )
    
    result = run_tryon_v2(req)
    assert result.status == "ok"
    assert Path(result.output_path).exists()
    print(f"✓ Modo overlay funcionó: {result.output_path}")
    
    # Test 2: En modo detección - debería detectar la camiseta naranja y reemplazarla
    req2 = TryOnRequest(
        image_path=str(person),
        garment_path=str(texture),
        output_path=str(output),
        garment_type="shirt",
        detect_and_replace_garment=True,
    )
    
    result2 = run_tryon_v2(req2)
    assert result2.status == "ok"
    assert Path(result2.output_path).exists()
    assert "garment_detection" in result2.meta.get("render", {}).get("mode", "")
    print(f"✓ Modo detección funcionó: {result2.output_path}")
    
    # Verifica que la salida cambió (la textura fue aplicada)
    output_arr = np.array(Image.open(output))
    assert output_arr.shape == person_arr.shape
    print(f"✓ Dimensiones correctas: {output_arr.shape}")
    
    # Compara cantidad de píxeles que cambieron
    diff = np.sum(np.abs(output_arr.astype(int) - person_arr.astype(int)))
    print(f"✓ Píxeles diferentes: {diff}")
    
    assert diff > 10000, f"Se esperaba más diferencia (mínimo 10000 píxeles), pero solo hay {diff}"
