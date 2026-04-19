from fase2_1.core.tryon.align import compute_scale


class _Point:
    def __init__(self, x: float):
        self.x = x


class _Landmarks:
    def __init__(self):
        self.landmark = [_Point(0.0) for _ in range(33)]
        self.landmark[11] = _Point(0.2)
        self.landmark[12] = _Point(0.8)
        self.landmark[23] = _Point(0.3)
        self.landmark[24] = _Point(0.7)


def test_compute_scale_garment_type_multiplier() -> None:
    lm = _Landmarks()
    base = compute_scale(lm, garment_type="shirt")
    dress = compute_scale(lm, garment_type="dress")
    pants = compute_scale(lm, garment_type="pants")

    assert dress > base
    assert pants < base
