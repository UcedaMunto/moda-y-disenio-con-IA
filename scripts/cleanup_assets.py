#!/usr/bin/env python3
"""
Cleanup script: Remove duplicate assets and keep only 1-2 useful variants per model/fabric.
Strategy:
- For fabrics: Keep only real fabric photos (webp, jpg, jpeg), remove Blender-generated maps duplicates
- For models: Keep all target_garment (shirt/skirt/pants) in both OBJ and FBX when available, 
  but remove _1, _2, _3 duplicates. Keep only 1 variant of non-target models.
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data").resolve()
TELAS_DIR = DATA_DIR / "raw" / "telas"
MODELS_DIR = DATA_DIR / "raw" / "models"

REAL_FABRIC_EXTS = {".webp", ".jpg", ".jpeg"}
BLENDER_MAP_KEYWORDS = {"ambient occlusion", "color map", "normal map", "curvature"}

def should_keep_fabric(filename: str) -> bool:
    """Keep only real fabric photos, not Blender-generated maps (unless they're unique first ones)."""
    lower = filename.lower()
    
    # Keep all real fabric photos
    if any(lower.endswith(ext) for ext in REAL_FABRIC_EXTS):
        return True
    
    # For PNG files: keep only if NOT a Blender map, OR if it's the first in its group (no suffix)
    if filename.lower().endswith(".png"):
        is_blender_map = any(kw in lower for kw in BLENDER_MAP_KEYWORDS)
        if is_blender_map:
            # Check if it has a numbered suffix (_1, _2, etc.)
            base = filename.rsplit(".", 1)[0]
            if base.endswith(("_1", "_2", "_3", "_4")):
                return False  # Remove duplicates
            else:
                return True  # Keep the first occurrence
        return True
    
    return True

def should_keep_model(filename: str) -> bool:
    """Keep target models, but deduplicate numbered variants.
    
    Strategy:
    - For all models: Remove _1, _2, _3, etc. suffixes (keep only base name)
    - For target garments: Keep both OBJ and FBX of the base model
    - For non-targets: Keep all formats of base model
    """
    lower = filename.lower()
    
    # Check if it has a numbered suffix
    base = filename.rsplit(".", 1)[0]
    if base.endswith(("_1", "_2", "_3", "_4", "_5")):
        return False  # Remove all numbered variants
    
    # Keep everything else (base models in all formats)
    return True

def cleanup_directory(directory: Path, filter_func) -> dict:
    """Remove files that don't pass the filter function."""
    if not directory.exists():
        return {"kept": 0, "removed": 0, "files": []}
    
    kept = []
    removed = []
    
    for item in sorted(directory.iterdir()):
        if not item.is_file():
            continue
        
        filename = item.name
        if filter_func(filename):
            kept.append(filename)
        else:
            removed.append(filename)
            item.unlink()
            print(f"  ✓ Removed: {filename}")
    
    return {
        "kept": len(kept),
        "removed": len(removed),
        "files_kept": kept,
    }

# Main cleanup
print("🧹 Cleaning up asset duplicates...\n")

print("📦 Fabrics (telas):")
telas_result = cleanup_directory(TELAS_DIR, should_keep_fabric)
print(f"  → Kept: {telas_result['kept']}, Removed: {telas_result['removed']}\n")

print("📦 Models (modelos):")
models_result = cleanup_directory(MODELS_DIR, should_keep_model)
print(f"  → Kept: {models_result['kept']}, Removed: {models_result['removed']}\n")

print("✅ Cleanup complete!")
print(f"Total files kept: {telas_result['kept'] + models_result['kept']}")
print(f"Total files removed: {telas_result['removed'] + models_result['removed']}")
