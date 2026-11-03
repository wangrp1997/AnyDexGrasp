"""Read-only official-code interface checks; not a grasp reproduction.

Runs the unmodified decision model, AST-selected official decision loss and
feature canonicalization. Synthetic tensors are used only for interfaces and
gradient/invariance checks. No trained weights, robot entrypoint, simulator,
package installation or environment mutation is used.
"""
from __future__ import annotations

import ast
import collections.abc
import copy
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT
OUT = ROOT / "outputs/runtime_audit"
OUT.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path, names: list[str], extra: dict | None = None) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    scope = {
        "__name__": "official_ast_extract",
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "copy": copy,
        "Dataset": Dataset,
        "container_abcs": collections.abc,
    }
    scope.update(extra or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main() -> None:
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261103)
    np.random.seed(20261103)
    report = {
        "scope": "official decision-network and feature-interface validation only",
        "official_source_revision": "c9c4a43df33e40860417c7e2dd02f5122d3b2da2",
        "training_model_kwargs": {"num_multifinger_type": 1, "num_multifinger_depth": 4, "num_two_finger_angle": 12, "num_two_finger_depth": 5},
        "not_grasp_reproduction": True,
        "inputs": "deterministic synthetic tensors; no scientific success-rate claim",
        "weights": "random initialization; no trained checkpoint loaded",
        "hardware_actions": "none; CPU tensors only; no robot entrypoint imported",
        "environment_modified": False,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_built": torch.version.cuda,
        "packages": {},
        "source_files": {},
        "checks": {},
    }
    for name in ["MinkowskiEngine", "graspnetAPI", "knn_pytorch", "pointnet2", "open3d", "numpy", "scipy"]:
        try:
            spec = importlib.util.find_spec(name)
            report["packages"][name] = str(spec.origin) if spec else None
        except Exception as exc:
            report["packages"][name] = {"error": str(exc)}

    paths = {
        "decision": SOURCE / "models/minkowski_graspnet.py",
        "representation": SOURCE / "models/minkowski_graspnet_single_point.py",
        "loss": SOURCE / "models/loss.py",
        "dataset": SOURCE / "dataset/multifinger_hand.py",
        "robot": SOURCE / "robot_allegro.py",
    }
    report["source_files"] = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for key, path in paths.items()}

    spec = importlib.util.spec_from_file_location("anydex_official_decision", paths["decision"])
    decision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decision)
    model = decision.MinkowskiGraspNetMultifingerType1(num_multifinger_type=1)
    batch = 8
    data = {
        "two_fingers_pose_angle_type": torch.arange(batch).view(-1, 1) % 12,
        "two_fingers_pose_depth_type": torch.arange(batch).view(-1, 1) % 5,
        "multifinger_pose_finger_type": torch.ones(batch, 1, dtype=torch.long),
        "multifinger_pose_depth_type": torch.arange(batch).view(-1, 1) % 4,
        "grasp_preds_features": torch.randn(batch, 480),
        "if_flip": torch.arange(batch).view(-1, 1) % 2,
        "result": torch.arange(batch).view(-1, 1) % 2,
    }
    model.train()
    pred, endpoints = model(dict(data))
    endpoints["result"] = data["result"]
    assert pred.shape == (batch, 1, 240)
    assert torch.isfinite(pred).all()
    criterion_cls = extract(paths["loss"], ["MultifingerType1Loss"])["MultifingerType1Loss"]
    criterion = criterion_cls(num_multifinger_type=1, train_type=1)
    loss, indicators, per_class = criterion(endpoints)
    idx = data["two_fingers_pose_depth_type"].view(-1) * 4 + data["multifinger_pose_depth_type"].view(-1)
    selected = pred.view(batch, -1)[torch.arange(batch), idx]
    reference_loss = F.mse_loss(selected, data["result"].view(-1).float())
    assert torch.equal(loss, reference_loss)
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert model.conv1.weight.grad.abs().sum() > 0
    report["checks"]["official_decision_forward_backward"] = {
        "passed": True,
        "input_shape": [batch, 480],
        "output_shape": list(pred.shape),
        "parameters": sum(param.numel() for param in model.parameters()),
        "loss": float(loss.detach()),
        "loss_formula": "MSE(selected depth output, real-grasp binary result); original code, synthetic labels here",
        "selected_output_indices": idx.tolist(),
        "finite_gradients": True,
        "optimizer_steps": 0,
    }

    inference_cls = extract(paths["representation"], ["MinkowskiGraspNetMultifingerType1Inference"])["MinkowskiGraspNetMultifingerType1Inference"]
    inference = inference_cls(input_num=480)
    inference.eval()
    with torch.no_grad():
        inference_pred, returned_features = inference(data["grasp_preds_features"])
    assert inference_pred.shape == (batch, 1, 20)
    assert torch.equal(returned_features, data["grasp_preds_features"])
    report["checks"]["official_inference_class"] = {"passed": True, "output_shape": list(inference_pred.shape)}
    try:
        inference.load_state_dict(model.state_dict())
    except RuntimeError as exc:
        report["checks"]["released_train_to_inference_shape_compatibility"] = {
            "passed": False,
            "expected_blocker_verified": True,
            "training_default_output_channels": 240,
            "inference_default_output_channels": 20,
            "original_exception": str(exc),
            "interpretation": "Released training entry defaults are not directly compatible with released robot inference class. Check published checkpoint architecture before changing source.",
        }
    else:
        raise AssertionError("Expected released default-head mismatch was not observed")

    dataset_cls = extract(paths["dataset"], ["MultifingerDataset"])["MultifingerDataset"]
    canonicalize = extract(paths["robot"], ["get_graspgroup_features"])["get_graspgroup_features"]
    dataset = object.__new__(dataset_cls)
    dataset.multifinger_type = "Allegro"
    dataset.data = []
    dataset.informations = {}
    rows = []
    expected = []
    for angle in range(12):
        for flip in [0, 1]:
            key = f"angle{angle}_flip{flip}"
            feature = np.arange(480, dtype=np.float32) + len(rows) * 1000
            info = {
                "two_fingers_pose_angle_type": angle,
                "two_fingers_pose_depth_type": angle % 5,
                "Allegro_pose_finger_type": 1,
                "Allegro_pose_depth_type": angle % 4,
                "if_flip": flip,
                "grasp_preds_features": feature.tolist(),
                "result": bool(flip),
            }
            dataset.data.append(key)
            dataset.informations[key] = info
            expected.append(dataset.__getitem__(len(rows))["grasp_preds_features"])
            row = np.zeros(240 + 480 + 512 * 3 + 6, dtype=np.float32)
            row[240:720] = feature
            row[-3] = 12 + angle * 2
            row[-2] = (angle % 5) / 100
            row[-1] = flip
            rows.append(row)
    result = canonicalize(np.stack(rows), sinput=None)
    expected = np.stack(expected)
    assert np.array_equal(result["grasp_preds_features"], expected)
    report["checks"]["training_vs_robot_feature_canonicalization"] = {
        "passed": True,
        "cases": len(rows),
        "coverage": "all 12 training angle types x both 180-degree flips",
        "assumed_angle_mapping": "robot 48-bin angle = 12 + 2 * training angle type",
        "feature_width": 480,
        "maximum_error": float(np.max(np.abs(result["grasp_preds_features"] - expected))),
        "boundary": "Synthetic records validate source convention only; downloaded dataset convention still needs verification.",
    }
    report["runtime_seconds"] = time.monotonic() - start
    (OUT / "runtime_interface_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (OUT / "runtime_interface_failure.txt").write_text(traceback.format_exc())
        raise
