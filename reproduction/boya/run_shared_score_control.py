#!/usr/bin/env python3
"""Frozen, object-grouped offline Boya scoring control on the existing435.

No new grasp candidates, CGR inference, simulator, external objects, or changes
to the old deployment are invoked. Historical trial-pool ranks are diagnostics,
not success rates for a new grasp policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_boya_decision as original
import shared_score_data as data_module
import shared_score_metrics as metric_module
from shared_score_model import (CONFIG, FAMILIES, SharedScorer, family_config,
                                fit_model, object_weights, predict)

DATASET = ROOT / "outputs/s40_stable_supplement_20261107_v2/training_delivery_v3/merged_dataset/manifest.json"
BASELINE = ROOT / "outputs/boya_fast_audit_20261111_v1/IMMUTABLE_BASELINE.json"
OLD_REPORTS = {
    "original_mse": ROOT / "outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json",
    "preactivation_mse": ROOT / "outputs/s40_preactivation_control_20261107_v1/TRAINING_RESULTS.json",
}
SCHEMA = "anydex_boya_shared_score_control_v1"
CONTINUOUS_COLUMNS = list(range(480)) + [494, 495]
GATES = {
    "cv_min_macro_auc": .60,
    "cv_min_gain_over_condition_linear": .05,
    "cv_min_same_observation_gain": .03,
    "cv_min_positive_folds": 3,
    "cv_bootstrap_gain_lower_must_exceed": 0.,
    "internal_min_gain_over_both_old_and_condition": .05,
    "internal_same_observation_auc_must_not_drop": True,
    "internal_mixed_pool_top_tie_must_not_drop": True,
    "cv_and_internal_macro_brier_must_not_exceed_object_prior": True,
}


def sha(path):
    return original.sha(path)


def save(path, value):
    original.save(path, value)


def read(path):
    return json.loads(Path(path).read_text())


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def freeze_sources():
    names = ["run_shared_score_control.py", "shared_score_data.py",
             "shared_score_metrics.py", "shared_score_model.py",
             "train_s40_code_recipe.py", "train_s40_decision.py",
             "train_boya_decision.py", "render_s40_observations.py"]
    paths = [HERE / name for name in names]
    paths += [original.MODEL_SOURCE, original.LOSS_SOURCE, original.SOLVER_SOURCE]
    return {str(p): sha(p) for p in paths}


def verify_hashes(mapping):
    failures = [p for p, h in mapping.items() if not Path(p).is_file() or sha(p) != h]
    if failures:
        raise ValueError("Frozen input/source changed: " + repr(failures[:5]))


def serialize_fold(fold):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fold.items()}


def prepare(output):
    if output.exists():
        raise ValueError("Fresh output directory required; never overwrite or resume a run")
    started = time.monotonic()
    manifest, a, inputs = data_module.load_shared_data(DATASET)
    immutable = read(BASELINE)["input_sha256"]
    verify_hashes(immutable)
    inputs.update(immutable)
    inputs[str(BASELINE)] = sha(BASELINE)
    folds = data_module.grouped_folds(a["mesh_sha256"], a["split"], 4, 20261111,
                                      declared_train_meshes=manifest["mesh_groups"]["train"])
    if len(a["labels"]) != 435 or sum(a["split"] == "train") != 364:
        raise ValueError("Original435/364/71 contract violated")
    sources = freeze_sources()
    protocol = {
        "schema": SCHEMA, "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_manifest": {"path": str(DATASET), "sha256": sha(DATASET)},
        "config": CONFIG, "families": list(FAMILIES), "continuous_columns": CONTINUOUS_COLUMNS,
        "fit_only_preprocessing": "per-fold train mean/std for CGR and two widths; onehots/flip unchanged",
        "features": "480 canonical + gesture4 + raw_depth5 + extra_depth4 + flip1 + widths2 =496",
        "fold_seed": 20261111, "folds": [serialize_fold(f) for f in folds],
        "all_trial_ids": a["trial_id"].tolist(),
        "labelled_groups": {split: {"records": int(sum(a["split"] == split)),
            "objects": len(set(a["object_id"][a["split"] == split])),
            "declared_objects": len(manifest["mesh_groups"][split])}
            for split in ("train", "internal_check")},
        "internal_trial_ids": a["trial_id"][a["split"] == "internal_check"].tolist(),
        "selection": "shared_linear vs mean-probability three-seed shared_mlp64; maximum CV macro object AUC, exact tie favors linear; selection saved BEFORE internal fits/metrics",
        "old_models": "Only forward on original71 internal records; never claim original32 training predictions are OOF",
        "gate": GATES,
        "metric_boundary": "Only historically executed finite pools; same-observation ranking separate; no claim of policy grasp success or unbiased external generalization",
        "internal_boundary": "Original8 repeatedly used during development; neither CV nor internal8 is a fresh external test",
        "bootstrap": "2000 paired object resamples, seed20261111; development uncertainty, post-selection optimism remains",
        "budget": {"folds": 4, "final32_fit_stages": 1, "trained_models": 25,
                   "optimizer_steps": 14500, "new_physics_steps": 0, "new_CGR_forwards": 0,
                   "external_objects": 0, "all40_refit": False},
        "input_sha256": inputs, "source_sha256": sources,
        "software": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
                     "device": "cpu", "torch_threads": 4, "deterministic": True},
        "prepare_wall_s": time.monotonic() - started,
    }
    output.mkdir(parents=True)
    save(output / "PROTOCOL.json", protocol)
    save(output / "PROGRESS.json", {"status": "prepared", "new_optimizer_steps": 0,
                                    "new_physics_steps": 0, "new_CGR_forwards": 0})
    return protocol


def metrics(a, ids, scores):
    args = (a["labels"][ids], scores)
    object_metrics = metric_module.evaluate_scores(*args, a["object_id"][ids], a["trial_id"][ids])
    observation_metrics = metric_module.evaluate_scores(*args, a["observation_group"][ids], a["trial_id"][ids])
    return {"object": object_metrics, "same_observation": observation_metrics}


def save_predictions(path, a, ids, scores):
    np.savez_compressed(path, indices=ids, trial_id=a["trial_id"][ids],
                        object_id=a["object_id"][ids], mesh_sha256=a["mesh_sha256"][ids],
                        observation_group=a["observation_group"][ids], labels=a["labels"][ids],
                        scores=np.asarray(scores, np.float64))
    return {"path": str(path), "sha256": sha(path)}


def run_split(a, fit, check, output, progress_callback):
    output.mkdir(parents=True)
    if set(a["mesh_sha256"][fit]) & set(a["mesh_sha256"][check]):
        raise ValueError("Fit/check mesh overlap")
    stats = data_module.fit_standardizer(a["all_features"], fit, CONTINUOUS_COLUMNS)
    xfit = data_module.transform(a["all_features"][fit], stats)
    xcheck = data_module.transform(a["all_features"][check], stats)
    save(output / "STANDARDIZER.json", {k: np.asarray(v).tolist() for k, v in stats.items()})
    np.savez_compressed(output / "STANDARDIZER.npz", **stats)
    split_report = {"fit_indices": fit.tolist(), "check_indices": check.tolist(),
                    "fit_trial_ids": a["trial_id"][fit].tolist(), "check_trial_ids": a["trial_id"][check].tolist(),
                    "fit_meshes": sorted(set(a["mesh_sha256"][fit])),
                    "check_meshes": sorted(set(a["mesh_sha256"][check])), "models": {}}
    save(output / "SPLIT.json", split_report)
    row_prior = float(a["labels"][fit].mean())
    object_prior = float(np.dot(object_weights(a["object_id"][fit]), a["labels"][fit]))
    predictions = {"row_prior": np.full(len(check), row_prior),
                   "object_prior": np.full(len(check), object_prior)}
    for family in FAMILIES:
        seeds = CONFIG["mlp_seeds"] if family == "shared_mlp64" else [CONFIG["linear_seed"]]
        seed_scores = []
        for seed in seeds:
            started = time.monotonic()
            model, history = fit_model(xfit, a["labels"][fit], a["object_id"][fit], family, seed)
            score = predict(model, xcheck)
            seed_scores.append(score)
            key = family + "_seed" + str(seed)
            checkpoint_path = output / (key + ".pt")
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"schema": SCHEMA, "family": family, "seed": seed,
                        "model_state_dict": state, "fit_trial_ids": a["trial_id"][fit].tolist(),
                        "fit_meshes": sorted(set(a["mesh_sha256"][fit])),
                        "config": family_config(family), "input_dim": 496,
                        "deployment_promoted": False}, checkpoint_path)
            loaded = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
            reloaded = SharedScorer(family)
            reloaded.load_state_dict(loaded["model_state_dict"], strict=True)
            if not np.array_equal(predict(reloaded, xcheck), score):
                raise ValueError("Checkpoint strict reload changed predictions")
            result = {"checkpoint": {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)},
                      "family": family, "seed": seed, "history": history,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "optimizer_steps": len(history), "strict_reload_exact": True,
                      "wall_s": time.monotonic() - started,
                      "fit_metrics": metrics(a, fit, predict(model, xfit)),
                      "check_metrics": metrics(a, check, score),
                      "predictions": save_predictions(output / (key + "_single_predictions.npz"), a, check, score)}
            split_report["models"][key] = result
            save(output / "SPLIT.json", split_report)
            progress_callback(len(history), key)
        predictions[family] = np.mean(seed_scores, axis=0)
        if family == "shared_mlp64":
            for seed, score in zip(seeds, seed_scores):
                predictions[family + "_seed" + str(seed)] = score
    split_report["metrics"] = {k: metrics(a, check, v) for k, v in predictions.items()}
    split_report["predictions"] = {k: save_predictions(output / (k + "_predictions.npz"), a, check, v)
                                    for k, v in predictions.items()}
    save(output / "SPLIT.json", split_report)
    return predictions, split_report


def old_predictions(a, ids):
    cls, _, _ = original.official_components()
    results = {}
    for name, path in OLD_REPORTS.items():
        report = read(path)
        score = np.full(len(ids), np.nan)
        for g, entry in report["internal_heads"].items():
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("Old checkpoint changed")
            checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=True)
            if set(checkpoint["fit_mesh_sha256"]) & set(a["mesh_sha256"][ids]):
                raise ValueError("Historical model fit leaked into original internal8")
            model = cls(**checkpoint["architecture_kwargs"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            rows = np.flatnonzero(a["gesture_ids"][ids] == int(g))
            with torch.no_grad():
                p = model(torch.from_numpy(a["features"][ids[rows]]))[0].reshape(-1, 20)
                score[rows] = p[np.arange(len(rows)), a["target_index"][ids[rows]]].numpy()
        if not np.isfinite(score).all():
            raise ValueError("Incomplete historical predictions")
        results[name] = score
    return results


def select_candidate(cv):
    linear = cv["shared_linear"]["object"]["macro_object_auc"]
    mlp = cv["shared_mlp64"]["object"]["macro_object_auc"]
    if linear is None or mlp is None:
        raise ValueError("No mixed-object CV discrimination metric")
    return "shared_mlp64" if mlp > linear else "shared_linear"


def gate_result(selected, cv, internal, folds, bootstrap):
    c, baseline = cv[selected], cv["condition_linear"]
    i = internal[selected]
    refs = [internal[k] for k in ("original_mse", "preactivation_mse", "condition_linear")]
    fold_gains = [f["metrics"][selected]["object"]["macro_object_auc"] -
                  f["metrics"]["condition_linear"]["object"]["macro_object_auc"] for f in folds]
    # Bootstrap field is normalized by the runner, retaining the full receipt.
    lower = bootstrap["gain_ci95"][0]
    checks = {
        "cv_auc_at_least_0_60": c["object"]["macro_object_auc"] >= .60,
        "cv_object_gain_at_least_0_05": c["object"]["macro_object_auc"] >= baseline["object"]["macro_object_auc"] + .05,
        "cv_same_observation_gain_at_least_0_03": c["same_observation"]["macro_object_auc"] >= baseline["same_observation"]["macro_object_auc"] + .03,
        "cv_positive_gain_in_3_of_4_folds": sum(v > 0 for v in fold_gains) >= 3,
        "cv_bootstrap_gain_lower_positive": lower > 0,
        "cv_macro_brier_no_worse_than_object_prior": c["object"]["macro_object_brier"] <= cv["object_prior"]["object"]["macro_object_brier"],
        "internal_object_auc_beats_each_reference_by_0_05": i["object"]["macro_object_auc"] >= max(r["object"]["macro_object_auc"] for r in refs) + .05,
        "internal_same_observation_auc_no_drop": i["same_observation"]["macro_object_auc"] >= max(r["same_observation"]["macro_object_auc"] for r in refs),
        "internal_mixed_pool_top_tie_no_drop": i["object"]["macro_mixed_top_tie_mean_label"] >= max(r["object"]["macro_mixed_top_tie_mean_label"] for r in refs),
        "internal_macro_brier_no_worse_than_object_prior": i["object"]["macro_object_brier"] <= internal["object_prior"]["object"]["macro_object_brier"],
    }
    return {"passed": all(checks.values()), "checks": checks, "cv_fold_gains": fold_gains,
            "decision": "eligible_for_separate_external_protocol" if all(checks.values()) else "stop_no_physical_promotion",
            "boundary": "Predeclared engineering screen on reused development data; not statistical proof of external improvement"}


def bootstrap_receipt(ref, new):
    # Explicit implementation also keeps the acceptance contract independent of
    # naming choices in the separately reviewed metrics helper.
    left = {r["object_id"]: r["pair_auc"] for r in ref["per_object"] if r["pair_auc"] is not None}
    right = {r["object_id"]: r["pair_auc"] for r in new["per_object"] if r["pair_auc"] is not None}
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("No shared mixed objects for bootstrap")
    d = np.array([right[o] - left[o] for o in ids])
    rng = np.random.default_rng(20261111)
    means = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    return {"objects": ids, "count": len(ids), "gain": float(d.mean()),
            "gain_ci95": np.quantile(means, [.025, .975]).tolist(), "seed": 20261111,
            "replicates": 2000, "wins": int(sum(d > 0)), "ties": int(sum(d == 0)), "losses": int(sum(d < 0)),
            "boundary": "Development paired object bootstrap; selected-model optimism remains"}


def run(output):
    protocol_path = output / "PROTOCOL.json"
    protocol = read(protocol_path)
    protocol_sha = sha(protocol_path)
    if protocol["schema"] != SCHEMA or read(output / "PROGRESS.json")["status"] != "prepared":
        raise ValueError("Only an untouched prepared run can start; no implicit resume")
    verify_hashes(protocol["source_sha256"])
    verify_hashes(protocol["input_sha256"])
    if protocol["config"] != CONFIG or protocol["gate"] != GATES:
        raise ValueError("Protocol configuration mismatch")
    manifest, a, _ = data_module.load_shared_data(DATASET)
    folds = data_module.grouped_folds(a["mesh_sha256"], a["split"], 4, 20261111,
                                      declared_train_meshes=manifest["mesh_groups"]["train"])
    if [serialize_fold(f) for f in folds] != protocol["folds"] or a["trial_id"].tolist() != protocol["all_trial_ids"]:
        raise ValueError("Frozen folds/rows changed")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    progress = {"status": "training_running", "new_optimizer_steps": 0, "trained_models": 0,
                "new_physics_steps": 0, "new_CGR_forwards": 0, "protocol_sha256": protocol_sha}
    save(output / "PROGRESS.json", progress)

    def completed(steps, name):
        progress["new_optimizer_steps"] += steps
        progress["trained_models"] += 1
        progress["last_model"] = name
        progress["wall_s"] = time.monotonic() - start
        save(output / "PROGRESS.json", progress)
        print(json.dumps(progress), flush=True)

    try:
        fit = np.flatnonzero(a["split"] == "train")
        check = np.flatnonzero(a["split"] == "internal_check")
        oof = {}
        fold_reports = []
        coverage = np.zeros(len(a["labels"]), int)
        for k, fold in enumerate(folds):
            train_ids = np.asarray(fold["fit_indices"], int)
            check_ids = np.asarray(fold["check_indices"], int)
            predictions, report = run_split(a, train_ids, check_ids, output / f"cv_fold{k}", completed)
            fold_reports.append(report)
            coverage[check_ids] += 1
            for name, scores in predictions.items():
                oof.setdefault(name, np.full(len(a["labels"]), np.nan))[check_ids] = scores
        if not (coverage[fit] == 1).all() or coverage[check].any():
            raise ValueError("OOF coverage duplicated/missing or internal8 used")
        cv = {k: metrics(a, fit, v[fit]) for k, v in oof.items()}
        cv_refs = {k: save_predictions(output / (k + "_oof.npz"), a, fit, v[fit]) for k, v in oof.items()}
        save(output / "CV_RESULTS.json", {"metrics": cv, "predictions": cv_refs,
                                           "coverage": coverage.tolist(), "fold_count": 4})
        selected = select_candidate(cv)
        selection = {"selected": selected, "selection_scope": "original32 object-grouped CV only",
                     "cv_results_sha256": sha(output / "CV_RESULTS.json"),
                     "internal_predictions_computed": False, "created_utc": datetime.now(timezone.utc).isoformat()}
        save(output / "SELECTION.json", selection)
        selection_sha = sha(output / "SELECTION.json")
        predictions, internal_report = run_split(a, fit, check, output / "internal8", completed)
        predictions.update(old_predictions(a, check))
        internal = {k: metrics(a, check, v) for k, v in predictions.items()}
        internal_refs = {k: save_predictions(output / (k + "_internal8.npz"), a, check, v) for k, v in predictions.items()}
        save(output / "INTERNAL_RESULTS.json", {"metrics": internal, "predictions": internal_refs,
                                                 "boundary": protocol["internal_boundary"]})
        bootstrap = bootstrap_receipt(cv["condition_linear"]["object"], cv[selected]["object"])
        gate = gate_result(selected, cv, internal, fold_reports, bootstrap)
        if progress["new_optimizer_steps"] != 14500 or progress["trained_models"] != 25:
            raise ValueError("Fixed training budget changed")
        verify_hashes(protocol["input_sha256"])
        verify_hashes(protocol["source_sha256"])
        if sha(protocol_path) != protocol_sha or sha(output / "SELECTION.json") != selection_sha:
            raise ValueError("Frozen protocol/selection changed")
        result = {"schema": SCHEMA, "status": "complete", "protocol_sha256": protocol_sha,
                  "selected": selected, "selection_sha256": selection_sha,
                  "cv_metrics": cv, "internal_metrics": internal, "cv_bootstrap": bootstrap,
                  "gate": gate, "new_optimizer_steps": progress["new_optimizer_steps"],
                  "trained_models": progress["trained_models"], "new_physics_steps": 0,
                  "new_CGR_forwards": 0, "old_models_and_data_unchanged": True,
                  "source_freeze_unchanged": True, "deployment_promoted": False,
                  "wall_s": time.monotonic() - start,
                  "artifacts": {name: sha(output / name) for name in ["CV_RESULTS.json", "SELECTION.json", "INTERNAL_RESULTS.json"]}}
        save(output / "RESULTS.json", result)
        progress.update(status="complete", decision=gate["decision"], wall_s=result["wall_s"])
        save(output / "PROGRESS.json", progress)
        print(json.dumps({k: result[k] for k in ["status", "selected", "gate", "wall_s"]}, indent=2), flush=True)
        return result
    except BaseException:
        progress.update(status="failed_no_implicit_resume", error=traceback.format_exc(), wall_s=time.monotonic()-start)
        save(output / "PROGRESS.json", progress)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("New artifacts must stay inside AnyDexGrasp outputs")
    prepare(output) if args.stage == "prepare" else run(output)


if __name__ == "__main__":
    main()
