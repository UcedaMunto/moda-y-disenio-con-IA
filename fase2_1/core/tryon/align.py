def compute_scale(landmarks, garment_type: str = "other") -> float:
    """Calcula escala base de prenda usando hombros y cadera.

    El factor final se ajusta por categoria para reducir artefactos en prenda larga/corta.
    """
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12]
    left_hip = landmarks.landmark[23]
    right_hip = landmarks.landmark[24]

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    torso_width = abs(left_hip.x - right_hip.x)
    base = max(shoulder_width, torso_width) * 2.2

    multipliers = {
        "shirt": 1.0,
        "dress": 1.1,
        "pants": 0.95,
        "skirt": 1.05,
        "other": 1.0,
    }
    factor = multipliers.get((garment_type or "other").lower(), 1.0)
    return base * factor
