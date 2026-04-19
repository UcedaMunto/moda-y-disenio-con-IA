from __future__ import annotations

from pathlib import Path

from scripts.analyze_deepfashion_coverage import build_coverage_report, collect_ids


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_collect_ids_extracts_numeric_prefix(tmp_path: Path) -> None:
    root = tmp_path / "old"
    _touch(root / "pointcloud" / "100" / "100-pose01.ply")
    _touch(root / "pointcloud" / "100" / "100-pose02.ply")
    _touch(root / "pointcloud" / "200" / "200-anything.ply")
    _touch(root / "pointcloud" / "bad" / "foo.ply")

    ids = collect_ids(root, "pointcloud", ".ply")
    assert ids == {"100", "200"}


def test_build_coverage_report_counts_and_sets(tmp_path: Path) -> None:
    old_root = tmp_path / "data_deepfasho_antiguo"
    new_root = tmp_path / "data_deepfashon"

    _touch(old_root / "mesh" / "10" / "10-pose.obj")
    _touch(old_root / "pointcloud" / "10" / "10-pose.ply")
    _touch(old_root / "pointcloud" / "20" / "20-pose.ply")
    _touch(old_root / "pose" / "20" / "20-pose.pkl")
    _touch(old_root / "featureline" / "30" / "30-pose.ply")

    _touch(new_root / "point_cloud" / "20" / "20-pose.ply")
    _touch(new_root / "point_cloud" / "40" / "40-pose.ply")
    _touch(new_root / "DF3D_Featurelines" / "30" / "30-lines.ply")

    payload = build_coverage_report(old_root, new_root, include_samples=True, sample_limit=10)

    assert payload["status"] == "ok"
    assert payload["counts"]["old"]["mesh_obj"] == 1
    assert payload["counts"]["old"]["pointcloud_ply"] == 2
    assert payload["counts"]["old"]["pose_pkl"] == 1
    assert payload["counts"]["old"]["featureline_ply"] == 1
    assert payload["counts"]["old"]["union"] == 3  # 10,20,30

    assert payload["counts"]["new"]["point_cloud_ply"] == 2
    assert payload["counts"]["new"]["df3d_featurelines_ply"] == 1
    assert payload["counts"]["new"]["union"] == 3  # 20,30,40

    assert payload["counts"]["shared_union_ids"] == 2  # 20,30
    assert payload["counts"]["old_only_union_ids"] == 1  # 10
    assert payload["counts"]["new_only_union_ids"] == 1  # 40

    assert payload["samples"]["shared_ids"] == ["20", "30"]
    assert payload["samples"]["old_only_ids"] == ["10"]
    assert payload["samples"]["new_only_ids"] == ["40"]
