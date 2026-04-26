from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib
import importlib.util
import platform
import shutil
import subprocess
import sys


def _run_command(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": "command-not-found", "returncode": 127}

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def _detect_linux_gpus() -> list[dict[str, str]]:
    lspci_path = shutil.which("lspci")
    if not lspci_path:
        return []

    result = _run_command([lspci_path])
    if not result["ok"] and not result["stdout"]:
        return []

    matches: list[dict[str, str]] = []
    for raw_line in result["stdout"].splitlines():
        line = raw_line.strip()
        lower_line = line.lower()
        if not any(token in lower_line for token in ("vga", "3d controller", "display")):
            continue
        vendor = "amd" if any(token in lower_line for token in ("amd", "advanced micro devices", "radeon")) else "other"
        matches.append({"vendor": vendor, "description": line})
    return matches


def _collect_torch_status() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "hip_version": None,
            "device_name": None,
            "error": "torch-not-installed",
        }

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - defensive import path
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "hip_version": None,
            "device_name": None,
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "installed": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": device_name,
        "error": None,
    }


@dataclass(frozen=True)
class CompatibilityVerdict:
    local_gpu_ready: bool
    api_cpu_ready: bool
    recommended_mode: str
    reason: str


def build_compatibility_report(project_root: Path | None = None) -> dict:
    gpu_devices = _detect_linux_gpus()
    amd_devices = [device for device in gpu_devices if device["vendor"] == "amd"]
    rocminfo_path = shutil.which("rocminfo")
    hipconfig_path = shutil.which("hipconfig")
    torch_status = _collect_torch_status()

    rocm_runtime_ready = bool(rocminfo_path or hipconfig_path)
    torch_rocm_ready = bool(torch_status["installed"] and torch_status["hip_version"] and torch_status["cuda_available"])
    local_gpu_ready = bool(amd_devices and rocm_runtime_ready and torch_rocm_ready)

    if local_gpu_ready:
        verdict = CompatibilityVerdict(
            local_gpu_ready=True,
            api_cpu_ready=True,
            recommended_mode="local_gpu_or_api",
            reason="AMD + ROCm + PyTorch HIP detectados.",
        )
    elif amd_devices:
        missing_parts = []
        if not rocm_runtime_ready:
            missing_parts.append("ROCm runtime")
        if not torch_status["installed"]:
            missing_parts.append("PyTorch")
        elif not torch_status["hip_version"]:
            missing_parts.append("PyTorch con HIP/ROCm")
        elif not torch_status["cuda_available"]:
            missing_parts.append("backend GPU disponible en torch")

        verdict = CompatibilityVerdict(
            local_gpu_ready=False,
            api_cpu_ready=True,
            recommended_mode="api_cpu_only",
            reason="Falta " + ", ".join(missing_parts) + " para inferencia local en AMD.",
        )
    else:
        verdict = CompatibilityVerdict(
            local_gpu_ready=False,
            api_cpu_ready=True,
            recommended_mode="api_cpu_only",
            reason="No se detecto una GPU AMD compatible en el host.",
        )

    return {
        "host": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "project_root": str(project_root) if project_root else None,
        },
        "gpu": {
            "devices": gpu_devices,
            "amd_detected": bool(amd_devices),
        },
        "rocm": {
            "rocminfo_path": rocminfo_path,
            "hipconfig_path": hipconfig_path,
            "runtime_ready": rocm_runtime_ready,
        },
        "torch": torch_status,
        "verdict": asdict(verdict),
    }