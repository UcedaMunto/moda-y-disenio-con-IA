from fase2_1.core.tryon.contract import build_landmarks_contract, build_transform_contract


class _Point:
    def __init__(self, x: float, y: float, z: float, v: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = v


class _Landmarks:
    def __init__(self):
        self.landmark = [_Point(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
        self.landmark[11] = _Point(0.1, 0.2, 0.0, 0.9)
        self.landmark[12] = _Point(0.9, 0.2, 0.0, 0.9)
        self.landmark[23] = _Point(0.2, 0.8, 0.0, 0.8)
        self.landmark[24] = _Point(0.8, 0.8, 0.0, 0.8)


def test_build_landmarks_contract_shape() -> None:
    contract = build_landmarks_contract(_Landmarks())
    assert contract["version"] == "1.0"
    assert contract["space"] == "image_normalized"
    assert "left_shoulder" in contract["points"]
    assert "right_hip" in contract["points"]


def test_build_transform_contract_shape() -> None:
    transform = build_transform_contract(scale=1.5, garment_type="dress")
    assert transform["version"] == "1.0"
    assert transform["garment_type"] == "dress"
    assert transform["scale"] == 1.5
    assert transform["translation_norm"]["x"] == 0.0
