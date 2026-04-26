from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Fase23Settings:
    project_root: Path
    phase_root: Path
    vendor_root: Path
    opentryon_root: Path
    opentryon_repo_url: str
    opentryon_ref: str

    @classmethod
    def discover(cls, project_root: Path | None = None) -> "Fase23Settings":
        phase_root = Path(__file__).resolve().parents[1]
        resolved_project_root = project_root or phase_root.parent
        vendor_root = phase_root / "vendor"
        opentryon_root = vendor_root / "opentryon"
        return cls(
            project_root=resolved_project_root,
            phase_root=phase_root,
            vendor_root=vendor_root,
            opentryon_root=opentryon_root,
            opentryon_repo_url=os.getenv(
                "FASE23_OPENTRYON_REPO_URL",
                "https://github.com/tryonlabs/opentryon.git",
            ),
            opentryon_ref=os.getenv("FASE23_OPENTRYON_REF", "main"),
        )