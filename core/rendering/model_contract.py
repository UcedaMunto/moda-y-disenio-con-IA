from pathlib import Path

SUPPORTED_MODEL_EXTENSIONS = {".obj", ".glb", ".gltf", ".fbx"}


def validate_model_contract(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported model format: {suffix}. "
            f"Supported: {sorted(SUPPORTED_MODEL_EXTENSIONS)}"
        )

    uv_status = "unknown"
    uv_message = "UV validation skipped (trimesh unavailable or non-mesh payload)."

    try:
        import trimesh

        loaded = trimesh.load(model_path, force="mesh")
        uv = getattr(getattr(loaded, "visual", None), "uv", None)
        if uv is not None and len(uv) > 0:
            uv_status = "ok"
            uv_message = "UV coordinates detected."
        else:
            uv_status = "missing"
            uv_message = "Model has no UV coordinates; texture projection may fail."
    except Exception:
        # Validation remains soft to keep compatibility across environments.
        pass

    return {
        "model_path": str(path),
        "format": suffix,
        "uv_status": uv_status,
        "uv_message": uv_message,
    }
