"""Bound AnyDex-Boya trial records; NumPy only, no robot/Isaac imports.

Integration: prepare_binding -> write_bound_commands -> execute those commands ->
evaluation.json -> collect_record. The executor must echo trial_binding_sha256,
commands_sha256, completed_phases, total_physics_steps and complete/task_success.
A missing artifact/aborted run is never a negative grasp label. Existing unrelated
CanonDex trials cannot be retroactively claimed as bound AnyDex training data.

Canonicalization uses only two AST-extracted original functions. Their hardware
module is never imported. Raw features have 2261 columns, before appending flip.
Explicit if_flip must describe the rotation actually executed; None selects the
original DH3 flip policy. Store canonical features directly for a new Boya reader;
do not run the legacy dataset's angle conversion/canonicalization a second time.
"""
from __future__ import annotations
import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np

SCHEMA = "anydex_boya_bound_trial_v1"


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def array_ref(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "c_order_bytes_sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def official_functions(source: str | Path) -> dict:
    """Extract original definitions only; do not execute imports/top-level code."""
    source = Path(source)
    names = {"flip_ggarray", "get_graspgroup_features"}
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("Expected the two original canonicalization functions")
    scope = {"np": np, "copy": copy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def canonical_candidate(row15, raw_features2261, functions, *, if_flip=None) -> dict:
    row = np.asarray(row15)
    raw = np.asarray(raw_features2261)
    if row.shape != (15,) or raw.shape != (2261,) or not np.isfinite(row).all() or not np.isfinite(raw).all():
        raise ValueError("Finite raw row15 and original unappended feature row2261 required")
    if not np.isclose(row[2], raw[-1], atol=1e-8, rtol=0):
        raise ValueError("Candidate depth and paired raw feature depth disagree")
    # GraspNet 17 columns: score,width,height,depth,R9,t3,object_id.
    gg = np.concatenate((row[:2], np.array([.03], dtype=row.dtype), row[2:],
                         np.array([-1], dtype=row.dtype)))[None].copy()
    original_gg, original_flips = functions["flip_ggarray"](gg.copy())
    if if_flip is not None and type(if_flip) is not bool:
        raise TypeError("if_flip must be the actual executed bool, or None for original policy")
    actual_flip = bool(original_flips[0]) if if_flip is None else if_flip
    if actual_flip == bool(original_flips[0]):
        gg = original_gg
    elif actual_flip:
        gg[:, 4:13].reshape(-1, 3, 3)[:, :, 1:3] *= -1
    # np.c_ mirrors robot_dh3.py; original function modifies its CGR slice in place.
    feature_input = np.c_[raw[None].copy(), [actual_flip]]
    canonical = functions["get_graspgroup_features"](feature_input, None)
    cgr = canonical["grasp_preds_features"][0].copy()
    angle_id = int(canonical["grasp_angles"][0])
    depth_id = int(canonical["grasp_depths"][0])
    if not (0 <= angle_id < 24 and 0 <= depth_id < 5) or cgr.shape != (480,):
        raise ValueError("Unexpected original 48-angle/5-depth parser convention")
    return {"candidate_gg17_after_flip": gg[0].copy(),
            "features2262_after_original_function": feature_input[0].copy(),
            "canonical_CGR480": cgr, "if_flip": actual_flip,
            "official_policy_if_flip": bool(original_flips[0]),
            "two_fingers_pose_angle_type": angle_id,
            "two_fingers_pose_depth_type": depth_id}


def prepare_binding(source_npz, row_index, official_source, *, gesture_id,
                    aperture_id, aperture_sha256, calibrated_width_m,
                    extra_depth_index, execution_spec: Mapping, if_flip=None) -> dict:
    """Call before execution; execution_spec holds q/poses/controller/source refs."""
    source_npz, official_source = Path(source_npz).resolve(), Path(official_source).resolve()
    if type(row_index) is not int or row_index < 0:
        raise ValueError("Nonnegative source NPZ row index required")
    if type(extra_depth_index) is not int or not 0 <= extra_depth_index < 4:
        raise ValueError("Original extra depth index must be 0..3 (0/10/20/30mm)")
    if type(gesture_id) is not int or gesture_id < 1:
        raise ValueError("Explicit positive Boya gesture_id required")
    if not aperture_id or len(aperture_sha256) != 64 or calibrated_width_m <= 0 or not execution_spec:
        raise ValueError("Bind aperture identity/SHA/width and actual execution specification")
    source_digest = file_sha(source_npz)
    with np.load(source_npz, allow_pickle=False) as z:
        candidates, features = z["candidates"], z["candidate_features"]
        if candidates.ndim != 2 or candidates.shape[1] != 15 or features.shape != (len(candidates), 2261):
            raise ValueError("Expected original paired candidates Nx15/features Nx2261")
        row, raw = candidates[row_index].copy(), features[row_index].copy()
    if file_sha(source_npz) != source_digest:
        raise ValueError("Source NPZ changed during read")
    info = canonical_candidate(row, raw, official_functions(official_source), if_flip=if_flip)
    extra_m = .01 * extra_depth_index
    binding = {"schema": SCHEMA, "source_npz": str(source_npz), "source_npz_sha256": source_digest,
        "source_row_index": row_index, "source_row15": row.tolist(), "source_row15_identity": array_ref(row),
        "candidate_features_raw2261": raw.tolist(), "candidate_features_raw2261_identity": array_ref(raw),
        "canonical_CGR480": info["canonical_CGR480"].tolist(),
        "canonical_CGR480_identity": array_ref(info["canonical_CGR480"]),
        "official_canonical_source": str(official_source), "official_canonical_source_sha256": file_sha(official_source),
        "canonicalization_functions": ["flip_ggarray", "get_graspgroup_features"],
        "canonicalization_applied_once": True,
        "if_flip": info["if_flip"], "official_policy_if_flip": info["official_policy_if_flip"],
        "executed_camera_cgr_after_flip": info["candidate_gg17_after_flip"][4:16].tolist(),
        "two_fingers_pose_angle_type": info["two_fingers_pose_angle_type"],
        "two_fingers_pose_depth_type": info["two_fingers_pose_depth_type"],
        "multifinger_pose_finger_type": gesture_id, "multifinger_pose_depth_type": extra_depth_index,
        "aperture": {"id": str(aperture_id), "sha256": aperture_sha256, "actual_ref_width_m": float(calibrated_width_m)},
        "raw_width_m": float(row[1]), "raw_depth_m": float(row[2]),
        "extra_depth_m": extra_m, "total_depth_m": float(row[2]) + extra_m,
        "execution_spec": dict(execution_spec),
        "label_scope": "Boya IsaacSim declared execution outcome; not paper hardware task success"}
    binding["trial_binding_sha256"] = json_sha(binding)
    return binding


def check_binding(binding: Mapping) -> None:
    expected = binding["trial_binding_sha256"]
    if json_sha({k: v for k, v in binding.items() if k != "trial_binding_sha256"}) != expected:
        raise ValueError("Trial binding was edited")
    for path_key, sha_key in [("source_npz", "source_npz_sha256"),
                              ("official_canonical_source", "official_canonical_source_sha256")]:
        if file_sha(binding[path_key]) != binding[sha_key]:
            raise ValueError(f"Source changed: {path_key}")


def write_bound_commands(path, arrays: Mapping, binding: Mapping) -> str:
    """Prepare NPZ before execution. Executor must read this exact file/SHA."""
    check_binding(binding)
    if "trial_binding_sha256" in arrays:
        raise ValueError("Reserved command metadata key")
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"Pickled/object command array forbidden: {key}")
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays, trial_binding_sha256=np.asarray(binding["trial_binding_sha256"]))
    return file_sha(path)


def execution_label(evaluation: Mapping) -> tuple[int | None, str]:
    """Conservative, explicit new Isaac evaluator contract; no inferred failure."""
    steps = evaluation.get("total_physics_steps")
    if type(steps) is not int or steps <= 0:
        return None, "zero_step_or_missing_step_evidence"
    if evaluation.get("infrastructure_error") or evaluation.get("error") or evaluation.get("exception") or evaluation.get("errors"):
        return None, "infrastructure_error"
    if evaluation.get("complete") is not True:
        return None, "incomplete_execution"
    if "lift" not in evaluation.get("completed_phases", []):
        return None, "lift_phase_not_completed"
    if evaluation.get("label_eligible") is not True:
        return None, "executor_did_not_confirm_label_eligibility"
    result = evaluation.get("task_success")
    if type(result) is not bool:
        return None, "missing_boolean_task_outcome"
    return int(result), "executed_lift_success" if result else "executed_lift_failure"


def collect_record(binding: Mapping, evaluation_json, commands_npz) -> dict:
    check_binding(binding)
    evaluation_json, commands_npz = Path(evaluation_json).resolve(), Path(commands_npz).resolve()
    evaluation = json.loads(evaluation_json.read_text())
    commands_sha = file_sha(commands_npz)
    with np.load(commands_npz, allow_pickle=False) as z:
        command_binding = str(z["trial_binding_sha256"].item())
    digest = binding["trial_binding_sha256"]
    if command_binding != digest or evaluation.get("trial_binding_sha256") != digest:
        raise ValueError("Candidate/gesture/aperture binding not echoed by actual commands and executor")
    if evaluation.get("commands_sha256") != commands_sha:
        raise ValueError("Evaluation does not bind the actual commands file")
    label, reason = execution_label(evaluation)
    return {"schema": SCHEMA, "binding": dict(binding),
            "execution_artifacts": {"evaluation_json": str(evaluation_json),
                "evaluation_sha256": file_sha(evaluation_json), "commands_npz": str(commands_npz),
                "commands_sha256": commands_sha},
            "total_physics_steps": evaluation.get("total_physics_steps"),
            "completed_phases": evaluation.get("completed_phases"),
            "result": label, "label_eligible": label is not None, "outcome_status": reason,
            "backend": evaluation.get("backend", evaluation.get("engine")),
            "task_success_definition": evaluation.get("success_definition"),
            "numerical_diagnostics": evaluation.get("numerical_diagnostics"),
            "record_builder_source": str(Path(__file__).resolve()),
            "record_builder_sha256": file_sha(__file__)}


def verify_existing(source_npz, predecision_npz, official_source) -> dict:
    """Read-only comparison to previously saved original predecision artifacts."""
    fns = official_functions(official_source)
    with np.load(source_npz, allow_pickle=False) as z:
        rows, raw = z["candidates"], z["candidate_features"]
    def key(gg, feature):
        return (np.asarray(gg, dtype=np.float32).tobytes(), np.asarray(feature, dtype=np.float32).tobytes())
    expected = collections.defaultdict(list)
    for row, feature in zip(rows, raw):
        info = canonical_candidate(row, feature, fns)
        expected[key(info["candidate_gg17_after_flip"], info["features2262_after_original_function"])].append(info["canonical_CGR480"])
    with np.load(predecision_npz, allow_pickle=False) as z:
        gg, fs, canonical = z["ggarray"], z["grasp_features"], z["canonical_CGR"]
    for row, feature, cgr in zip(gg, fs, canonical):
        bucket = expected[key(row, feature)]
        if not bucket or not any(np.array_equal(cgr, v) for v in bucket):
            raise AssertionError("Original stored canonical CGR differs from paired source reconstruction")
    base = dict(total_physics_steps=10, complete=True, completed_phases=["close", "lift"],
                label_eligible=True, task_success=False)
    assert execution_label(base) == (0, "executed_lift_failure")
    assert execution_label({**base, "task_success": True})[0] == 1
    for change in [dict(total_physics_steps=0), dict(infrastructure_error="error"), dict(complete=False),
                   dict(completed_phases=["close"]), dict(label_eligible=False), dict(task_success=None)]:
        assert execution_label({**base, **change})[0] is None
    return {"canonical_rows_exact": len(gg), "raw_rows": len(rows), "label_guard_cases": 8,
            "source_npz_sha256": file_sha(source_npz), "predecision_sha256": file_sha(predecision_npz),
            "official_source_sha256": file_sha(official_source), "new_physics_steps": 0,
            "new_training_steps": 0, "new_network_forwards": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-existing", action="store_true", required=True)
    p.add_argument("--source-npz", type=Path, required=True)
    p.add_argument("--predecision-npz", type=Path, required=True)
    p.add_argument("--official-source", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(verify_existing(a.source_npz, a.predecision_npz, a.official_source), indent=2))
