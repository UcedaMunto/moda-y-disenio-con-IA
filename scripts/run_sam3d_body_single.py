#!/usr/bin/env python3
"""Run SAM 3D Body on one image and export mesh artifacts.

This script is intended to be called from the main API endpoint for Fase 2.3.
It requires a Python environment where SAM 3D Body dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM 3D Body single-image inference")
    parser.add_argument("--repo-root", default=".", help="Root path of this repository")
    parser.add_argument("--sam3d-root", default="sam-3d-body", help="Path to sam-3d-body clone")
    parser.add_argument("--image-path", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output artifact directory")
    parser.add_argument("--checkpoint-path", required=True, help="SAM 3D Body checkpoint path")
    parser.add_argument("--mhr-path", required=True, help="MHR asset path")
    parser.add_argument("--detector-name", default="", help="Optional detector name (vitdet/sam3)")
    parser.add_argument("--bbox-thresh", type=float, default=0.8, help="Detector bbox threshold")
    parser.add_argument("--use-mask", action="store_true", help="Use mask-conditioned inference")
    parser.add_argument(
        "--export-glb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export GLB alongside PLY meshes",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sam3d_root = _resolve(repo_root, args.sam3d_root)
    image_path = _resolve(repo_root, args.image_path)
    output_dir = _resolve(repo_root, args.output_dir)
    checkpoint_path = _resolve(repo_root, args.checkpoint_path)
    mhr_path = _resolve(repo_root, args.mhr_path)

    if not sam3d_root.exists():
        raise FileNotFoundError(f"sam-3d-body root not found: {sam3d_root}")
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not mhr_path.exists():
        raise FileNotFoundError(f"mhr path not found: {mhr_path}")

    # Ensure imports resolve from the external repository.
    sys.path.insert(0, str(sam3d_root))

    import cv2
    import numpy as np
    import torch
    import trimesh

    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    human_detector = None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        human_detector = HumanDetector(name=args.detector_name, device=device, path="")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        str(checkpoint_path),
        device=device,
        mhr_path=str(mhr_path),
    )

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = estimator.process_one_image(
        str(image_path),
        bbox_thr=float(args.bbox_thresh),
        use_mask=bool(args.use_mask),
    )

    mesh_paths: list[str] = []
    glb_paths: list[str] = []
    base_name = image_path.stem
    for idx, out in enumerate(outputs):
        vertices = out.get("pred_vertices")
        if vertices is None:
            continue
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=estimator.faces.copy(), process=False)
        mesh_path = output_dir / f"{base_name}_mesh_{idx:03d}.ply"
        mesh.export(mesh_path)
        mesh_paths.append(str(mesh_path))
        if args.export_glb:
            glb_path = output_dir / f"{base_name}_mesh_{idx:03d}.glb"
            mesh.export(glb_path)
            glb_paths.append(str(glb_path))

        meta = {
            "bbox": np.asarray(out.get("bbox", [])).tolist(),
            "focal_length": float(out.get("focal_length", 0.0)),
            "pred_cam_t": np.asarray(out.get("pred_cam_t", [])).tolist(),
        }
        (output_dir / f"{base_name}_mesh_{idx:03d}.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

    overlay_path = output_dir / f"{base_name}_overlay.jpg"
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is not None:
        try:
            from tools.vis_utils import visualize_sample_together

            rendered = visualize_sample_together(img_bgr, outputs, estimator.faces)
            cv2.imwrite(str(overlay_path), rendered.astype(np.uint8))
        except Exception:
            cv2.imwrite(str(overlay_path), img_bgr)

    payload = {
        "status": "ok",
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "mesh_paths": mesh_paths,
        "glb_paths": glb_paths,
        "overlay_path": str(overlay_path) if overlay_path.exists() else None,
        "n_people": len(outputs),
        "device": str(device),
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
