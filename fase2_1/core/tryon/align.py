def compute_scale(landmarks) -> float:
    """Calcula escala base de prenda usando hombros y cadera."""
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12]
    left_hip = landmarks.landmark[23]
    right_hip = landmarks.landmark[24]

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    torso_width = abs(left_hip.x - right_hip.x)
    return max(shoulder_width, torso_width) * 2.2
