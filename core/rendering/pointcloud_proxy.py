from __future__ import annotations

import hashlib
from pathlib import Path


def is_point_cloud_model_path(model_path: str) -> bool:
    lower = model_path.lower()
    return lower.endswith(".ply") or "/point_cloud/" in lower or "/pointcloud/" in lower


def build_point_cloud_proxy_mesh(
    model_path: str,
    cache_dir: str = "data/processed/pointcloud_meshes",
) -> tuple[str, dict]:
    """Build a lightweight mesh proxy from a point cloud PLY on demand.

    The proxy is cached by file path + size + mtime so repeated requests are fast.
    """
    src = Path(model_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not is_point_cloud_model_path(model_path):
        return str(src), {"converted": False, "reason": "not_point_cloud"}

    stat = src.stat()
    key_src = f"{src.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = out_dir / f"{src.stem}-{key}.obj"

    if out_mesh.exists() and out_mesh.stat().st_size > 0:
        return str(out_mesh), {
            "converted": True,
            "from": str(src),
            "to": str(out_mesh),
            "cache_hit": True,
            "method": "convex_hull",
        }

    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError("trimesh no disponible para convertir point cloud") from exc

    loaded = trimesh.load(str(src), process=False)

    # Accept both PointCloud and Trimesh payloads; convert to a renderable Trimesh.
    mesh = None
    if isinstance(loaded, trimesh.points.PointCloud):
        if len(loaded.vertices) < 4:
            raise RuntimeError("Point cloud con muy pocos puntos para generar malla")
        mesh = loaded.convex_hull
    elif isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) == 0:
            if len(loaded.vertices) < 4:
                raise RuntimeError("Modelo sin caras y con muy pocos vertices")
            mesh = trimesh.points.PointCloud(loaded.vertices).convex_hull
        else:
            mesh = loaded
    else:
        # Scene or other container: try to merge meshes.
        if hasattr(loaded, "geometry") and loaded.geometry:
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if geoms:
                mesh = trimesh.util.concatenate(geoms)

    if mesh is None or len(mesh.faces) == 0:
        raise RuntimeError("No se pudo generar una malla valida desde la nube de puntos")

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    mesh.export(str(out_mesh))

    return str(out_mesh), {
        "converted": True,
        "from": str(src),
        "to": str(out_mesh),
        "cache_hit": False,
        "method": "convex_hull",
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }
