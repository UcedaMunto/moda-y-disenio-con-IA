import json
from pathlib import Path

from fase2_1.core.training.fit_regression.baseline import (
    extract_feature_vector,
    extract_target_scale,
    fit_ridge_regression,
    load_rows_from_jsonl,
    predict_scale,
    save_model,
)


def _row(scale: float, x_shift: float = 0.0, garment_type: str = "shirt") -> dict:
    return {
        "landmarks_contract": {
            "points": {
                "left_shoulder": {"x": 0.2 + x_shift, "y": 0.2, "visibility": 1.0},
                "right_shoulder": {"x": 0.8 + x_shift, "y": 0.2, "visibility": 1.0},
                "left_hip": {"x": 0.3 + x_shift, "y": 0.7, "visibility": 1.0},
                "right_hip": {"x": 0.7 + x_shift, "y": 0.7, "visibility": 1.0},
            }
        },
        "transform_contract": {
            "scale": scale,
            "garment_type": garment_type,
        },
    }


def test_extract_feature_vector_and_target() -> None:
    row = _row(scale=1.25, garment_type="pants")

    features = extract_feature_vector(row)
    target = extract_target_scale(row)

    assert len(features) == (4 * 3) + 5
    assert target == 1.25
    # Pants one-hot index should be active.
    assert features[-3] == 1.0


def test_fit_ridge_regression_and_predict() -> None:
    rows = [_row(scale=1.0, x_shift=0.0), _row(scale=1.1, x_shift=0.01), _row(scale=1.2, x_shift=0.02)]
    x = [extract_feature_vector(r) for r in rows]
    y = [extract_target_scale(r) for r in rows]

    model = fit_ridge_regression(x, y, l2=1e-4)
    pred = predict_scale(model, x[1])

    assert model["status"] == "ok"
    assert model["feature_dim"] == len(x[0])
    assert abs(pred - y[1]) < 0.05


def test_load_rows_and_save_model(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    model_path = tmp_path / "model.json"

    rows = [_row(scale=1.0), _row(scale=1.1)]
    dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_rows_from_jsonl(str(dataset))
    assert len(loaded) == 2

    model = fit_ridge_regression([extract_feature_vector(r) for r in loaded], [extract_target_scale(r) for r in loaded])
    saved = save_model(model, str(model_path))
    assert saved == str(model_path)
    assert model_path.exists()
