"""Read-only dependency and native-runtime probes; no installs or hardware control."""
from __future__ import annotations
import glob
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/runtime_audit"
OUT.mkdir(parents=True, exist_ok=True)
for sub in ["models", "pointnet2", "knn", "utils"]:
    sys.path.insert(0, str(ROOT / sub))
result = {"source_checkout": str(ROOT), "python_executable": sys.executable,
          "read_only_probe": True, "packages_installed": False,
          "environments_created": False, "imports": {}}
for module in ["minkowski_graspnet_single_point", "knn_modules", "pointnet2_utils"]:
    try:
        importlib.import_module(module)
        result["imports"][module] = {"imported": True}
    except Exception as exc:
        result["imports"][module] = {"imported": False, "error_type": type(exc).__name__,
                                   "error": str(exc), "traceback": traceback.format_exc()}
result["nvcc_on_path"] = shutil.which("nvcc")
patterns = [
    "/usr/local/cuda*/bin/nvcc", "/opt/cuda*/bin/nvcc", "/usr/bin/nvcc", "/usr/local/bin/nvcc",
    "/opt/nvidia/cuda*/bin/nvcc", "/mnt/public/cuda*/bin/nvcc",
    "/opt/conda/bin/python*", "/opt/conda/envs/*/bin/python*",
    "/home/*/miniconda*/bin/python*", "/home/*/miniconda*/envs/*/bin/python*",
    "/home/*/anaconda*/bin/python*", "/home/*/anaconda*/envs/*/bin/python*",
    "/home/*/.local/share/uv/python/*3.8*/bin/python*",
    "/home/*/.cache/uv/python/*3.8*/bin/python*", "/home/*/.pyenv/versions/3.8*/bin/python*",
    "/usr/bin/python3.8", "/usr/local/bin/python3.8",
    "/opt/nros-host/usr/local/cuda*/bin/nvcc", "/opt/nros-host/usr/bin/python3.8",
]
result["native_path_search"] = {pattern: glob.glob(pattern) for pattern in patterns}
result["bounded_site_package_search"] = {}
for pattern in [
    "/home/*/*/.venv/lib/python*/site-packages/MinkowskiEngine",
    "/home/*/*/.venv/lib/python*/site-packages/graspnetAPI",
    "/home/*/*/.venv/lib/python*/site-packages/pointnet2/_ext*.so",
    "/home/*/*/.venv/lib/python*/site-packages/knn_pytorch/*.so",
    "/home/*/*/.venv/lib/python*/site-packages/knn_pytorch*.so",
]:
    result["bounded_site_package_search"][pattern] = glob.glob(pattern)
result["extension_cache_directories"] = glob.glob("/home/*/.cache/torch_extensions")
cache_roots = [str(path) for path in [Path.home()/".cache/pip/wheels", Path.home()/".cache/uv"] if path.exists()]
if shutil.which("rg") and cache_roots:
    try:
        probe = subprocess.run(["rg", "--files", "-uu", *cache_roots], capture_output=True, text=True, timeout=30)
        import re
        result["cache_candidate_matches"] = [line for line in probe.stdout.splitlines()
            if re.search(r"(minkowskiengine|graspnetapi|pointnet2|knn_pytorch)",line,re.I)]
        result["cache_scan_returncode"] = probe.returncode
    except subprocess.TimeoutExpired:
        result["cache_scan_timeout_seconds"] = 30
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20)
    result["gpu_query"] = {"returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr}
except Exception as exc:
    result["gpu_query"] = {"error": str(exc)}
(OUT / "dependency_native_probes.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
