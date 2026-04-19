def filter_candidates(candidates: list[str], mode: str = "approve") -> list[str]:
    """Simple selector placeholder.

    In phase 0 this only keeps non-empty values when mode is approve.
    """
    if mode != "approve":
        return []
    return [item for item in candidates if item]
