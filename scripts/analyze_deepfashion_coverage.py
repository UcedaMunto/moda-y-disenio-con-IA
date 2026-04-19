from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ID_PATTERN = re.compile(r"([0-9]+)-")


def _extract_id_from_name(path: Path) -> str | None:
    match = ID_PATTERN.search(path.name)
    if not match:
        return None
    return match.group(1)


def collect_ids(base_dir: Path, subdir: str, suffix: str) -> set[str]:
    root = base_dir / subdir
    if not root.exists() or not root.is_dir():
        return set()

    out: set[str] = set()
    for path in sorted(root.rglob(f"*{suffix}")):
        if not path.is_file():
            continue
        item_id = _extract_id_from_name(path)
        if item_id:
            out.add(item_id)
    return out


def build_coverage_report(
    old_root: Path,
    new_root: Path,
    include_samples: bool = True,
    sample_limit: int = 25,
) -> dict:
    old_mesh = collect_ids(old_root, "mesh", ".obj")
    old_pointcloud = collect_ids(old_root, "pointcloud", ".ply")
    old_pose = collect_ids(old_root, "pose", ".pkl")
    old_featureline = collect_ids(old_root, "featureline", ".ply")

    new_pointcloud = collect_ids(new_root, "point_cloud", ".ply")
    new_featureline = collect_ids(new_root, "DF3D_Featurelines", ".ply")

    old_union = old_mesh | old_pointcloud | old_pose | old_featureline
    new_union = new_pointcloud | new_featureline

    shared = old_union & new_union
    old_only = old_union - new_union
    new_only = new_union - old_union

    report = {
        "status": "ok",
        "paths": {
            "old_root": str(old_root),
            "new_root": str(new_root),
        },
        "counts": {
            "old": {
                "mesh_obj": len(old_mesh),
                "pointcloud_ply": len(old_pointcloud),
                "pose_pkl": len(old_pose),
                "featureline_ply": len(old_featureline),
                "union": len(old_union),
            },
            "new": {
                "point_cloud_ply": len(new_pointcloud),
                "df3d_featurelines_ply": len(new_featureline),
                "union": len(new_union),
            },
            "shared_union_ids": len(shared),
            "old_only_union_ids": len(old_only),
            "new_only_union_ids": len(new_only),
        },
        "coverage": {
            "new_over_old_union_pct": round((len(shared) / len(old_union) * 100.0), 3) if old_union else 0.0,
            "old_over_new_union_pct": round((len(shared) / len(new_union) * 100.0), 3) if new_union else 0.0,
        },
        "consistency": {
            "old_mesh_vs_pointcloud_shared": len(old_mesh & old_pointcloud),
            "old_pointcloud_vs_pose_shared": len(old_pointcloud & old_pose),
            "new_pointcloud_vs_featureline_shared": len(new_pointcloud & new_featureline),
        },
    }

    if include_samples:
        report["samples"] = {
            "shared_ids": sorted(shared)[:sample_limit],
            "old_only_ids": sorted(old_only)[:sample_limit],
            "new_only_ids": sorted(new_only)[:sample_limit],
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza cobertura de IDs entre DeepFashion antiguo y nuevo")
    parser.add_argument("--old-root", default="data_deepfasho_antiguo")
    parser.add_argument("--new-root", default="data_deepfashon")
    parser.add_argument("--output", default="data/processed/deepfashion/coverage_report.json")
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=25)
    args = parser.parse_args()

    payload = build_coverage_report(
        old_root=Path(args.old_root),
        new_root=Path(args.new_root),
        include_samples=not args.no_samples,
        sample_limit=max(1, args.sample_limit),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
