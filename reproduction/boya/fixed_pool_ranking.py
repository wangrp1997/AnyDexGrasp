"""Label-blind ranking of a frozen, previously generated Boya candidate union.

No simulator, new CGR inference, optimizer, shared-score model or file writes.
The two old recipes' saved float32 outputs are authoritative for rankings;
strict CPU reloads independently check all 20 outputs on each canonical input.
This is a score-conditioned known-object pool, not an unfiltered generator test.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from evaluate_boya_decision import Reader, original_inference_class
from trial_record import array_ref, canonical_candidate, json_sha, official_functions

ROOT = Path(__file__).resolve().parents[2]
SCORERS = ("preactivation", "original_mse", "cgr")
GESTURES = (1, 6, 7, 8)
DATASET_SHA = "c85a284be680707a2b3bcaae7950e1981bd907a9e88870c434ced1c22f6e91bb"
MODEL_SOURCE_SHA = "1463ee5804ec72bafbfc0f469041425092817ff6e6b797923db77120e1167b26"
OFFICIAL_ROBOT_SHA = "7d227451761a429fff87f689e5dcb9c9e215da5ec3c5c594be1e16f3333fa797"
RECIPES = {
    "preactivation": {
        "training": "outputs/s40_preactivation_control_20261107_v1/TRAINING_RESULTS.json",
        "sha256": "68dbd71088928431f46d612d1458d4174a43dc54326fc9781c996667cbaf77ce",
        "diagnostic": "outputs/egad_external12_20261107_v1/diagnostic",
        "variant": "preactivation_selected_mse",
    },
    "original_mse": {
        "training": "outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json",
        "sha256": "447302d26c1e5cccc10f23ae897105a5d845abe661640cfc4ea5fa6d912ccac0",
        "diagnostic": "outputs/boya_original_recipe_alignment_20261108_v1/diagnostic",
        "variant": None,
    },
}
TIE_FIELDS = ("source_candidate_index", "gesture_id", "raw_depth_index",
              "extra_depth_index", "if_flip", "candidate_id")
FORWARD_ATOL = 2e-5
THRESHOLD = float(np.float32(.85))
BOUNDARY = (
    "Known-object diagnostic. Pool is the deduplicated union of two historical "
    "score-threshold/top-budget/geometry-admitted pools, not all CGR candidates. "
    "Historical outcomes are known but are not inputs to score_catalog. "
    "Any-success oracle applies only to this fixed pool and unchanged controller. "
    "No independent generalization or deployment promotion is established."
)


def catalog_rows(catalog):
    rows = catalog.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nonempty explicit candidates list required")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Candidate dictionary required")
        cid, oid = row.get("candidate_id"), row.get("object_id")
        if not isinstance(cid, str) or not cid or cid in seen or not isinstance(oid, str) or not oid:
            raise ValueError("Unique nonempty candidate_id and object_id required")
        seen.add(cid)
    return rows


def tie_key(row):
    return tuple(row[k] for k in TIE_FIELDS)


def rank_candidates(rows):
    """Rank without consulting catalog order, historical ranks, or outcomes."""
    for row in rows:
        if set(row["scores"]) != set(SCORERS) or not all(math.isfinite(float(v)) for v in row["scores"].values()):
            raise ValueError("Three finite scores required")
    result = []
    for oid in sorted({r["object_id"] for r in rows}):
        group = [r for r in rows if r["object_id"] == oid]
        rankings = {}
        for scorer in SCORERS:
            ordered = sorted(group, key=lambda r: (-r["scores"][scorer], tie_key(r)))
            rankings[scorer] = [r["candidate_id"] for r in ordered]
        result.append(dict(object_id=oid, candidate_count=len(group), rankings=rankings))
    return result


def _binding(item, reader, functions, raw_cache):
    path = Path(item.get("generation_json") or Path(item["candidate_dir"]) / "generation.json").resolve()
    generation = reader.read(path, item.get("generation_sha256"))
    records = generation.get("records", [])
    index = item.get("record_index", 0)
    if generation.get("complete") is not True or index != 0 or len(records) != 1:
        raise ValueError("One complete generated candidate per directory required")
    record = records[0]
    if record.get("geometry", {}).get("ok") is not True:
        raise ValueError("Candidate was not admitted by original hand geometry")
    candidates = path.parent / "candidates.npz"
    reader.freeze(candidates, generation["candidates_sha256"])
    raw_path = (Path(generation["source_frame"]) / "raw_camera_candidates.npz").resolve()
    expected = generation["input_sha256"].get(str(raw_path))
    if expected is None:
        raise ValueError("Generated candidate lacks its source raw-feature SHA")
    reader.freeze(raw_path, expected)
    if str(raw_path) not in raw_cache:
        with np.load(raw_path, allow_pickle=False) as z:
            raw_cache[str(raw_path)] = (z["candidates"].copy(), z["candidate_features"].copy())
    raw, features = raw_cache[str(raw_path)]
    source = record["source_candidate_index"]
    if type(source) is not int or not 0 <= source < len(raw) or features.shape != (len(raw), 2261):
        raise ValueError("Invalid source index/raw2261 binding")
    flip = record["if_flip"]
    info = canonical_candidate(raw[source], features[source], functions, if_flip=flip)
    canonical = np.asarray(info["canonical_CGR480"], dtype=np.float32)
    gesture, depth, extra = (record[k] for k in ("gesture_id", "raw_depth_index", "extra_depth_index"))
    if (gesture not in GESTURES or type(depth) is not int or type(extra) is not int
            or not 0 <= extra < 4 or depth != info["two_fingers_pose_depth_type"]
            or record["target_index"] != depth * 4 + extra
            or record["canonical_CGR480_identity"] != array_ref(canonical)
            or record["cgr_score"] != float(raw[source, 0])):
        raise ValueError("Generated candidate does not bind original canonical480/target/CGR")
    for key in ("source_candidate_index", "gesture_id", "raw_depth_index", "extra_depth_index", "if_flip"):
        if key in item and item[key] != record[key]:
            raise ValueError("Catalog and generation disagree: " + key)
    row = dict(candidate_id=item["candidate_id"], object_id=item["object_id"],
               generation_json=str(path), generation_sha256=reader.files[str(path)],
               candidate_npz=str(candidates), candidate_npz_sha256=reader.files[str(candidates)],
               source_npz=str(raw_path), source_npz_sha256=expected,
               source_row15_identity=array_ref(raw[source]),
               raw_features2261_identity=array_ref(features[source]),
               canonical_CGR480_identity=array_ref(canonical), canonicalization_applied_once=True,
               target_index=depth * 4 + extra,
               scores={"cgr": float(raw[source, 0])})
    row.update({k: record[k] for k in TIE_FIELDS if k != "candidate_id"})
    return row, canonical, record


def _saved_outputs(oid, scorer, training, reader, source_refs):
    directory = ROOT / RECIPES[scorer]["diagnostic"] / oid
    freeze = reader.read(directory / "RANKING_FREEZE.json")
    ref = freeze["training_results_ref"]
    if ref["sha256"] != RECIPES[scorer]["sha256"] or Path(ref["path"]).resolve() != ROOT / RECIPES[scorer]["training"]:
        raise ValueError("Saved scoring belongs to another training recipe")
    for path, expected in source_refs.items():
        if freeze["ranking_inputs_sha256"].get(path) != expected:
            raise ValueError("Saved head queries used different raw candidate features")
    for g in GESTURES:
        if freeze["models"][str(g)]["sha256"] != training["internal_heads"][str(g)]["sha256"]:
            raise ValueError("Saved scoring model differs from fixed old head")
    path = directory / "head_outputs.npz"
    reader.freeze(path, freeze["outputs_sha256"][str(path)])
    with np.load(path, allow_pickle=False) as z:
        values = z["outputs20"].copy()
        columns = [z[k] for k in ("source_indices", "gesture_ids", "raw_depth_indices", "if_flip")]
        keys = [(int(i), int(g), int(d), bool(f)) for i, g, d, f in zip(*columns)]
    if values.shape != (len(keys), 20) or values.dtype != np.float32 or not np.isfinite(values).all() or len(set(keys)) != len(keys):
        raise ValueError("Invalid saved original head outputs")
    return dict(zip(keys, values))


def score_catalog(catalog):
    """Return JSON-serializable scores; caller freezes this before new physics."""
    import torch
    rows_in = catalog_rows(catalog)
    reader = Reader()
    source_paths = (Path(__file__), ROOT / "reproduction/boya/trial_record.py",
                    ROOT / "reproduction/boya/evaluate_boya_decision.py",
                    ROOT / "robot_dh3.py", ROOT / "models/minkowski_graspnet_single_point.py")
    for path in source_paths[:3]:
        reader.freeze(path)
    official = ROOT / "robot_dh3.py"
    reader.freeze(official, OFFICIAL_ROBOT_SHA)
    functions = official_functions(official)
    rows, features, origins, raw_cache = [], [], [], {}
    for item in rows_in:
        row, feature, origin = _binding(item, reader, functions, raw_cache)
        rows.append(row); features.append(feature); origins.append(origin)
    source = ROOT / "models/minkowski_graspnet_single_point.py"
    reader.freeze(source, MODEL_SOURCE_SHA)
    cls = original_inference_class(source, torch)
    architecture = dict(input_num=480, num_multifinger_type=1, num_multifinger_depth=4,
                        two_finger_num_angle=1, two_finger_num_depth=5)
    models, audit, forward_batches = {}, {}, 0
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        for scorer, recipe in RECIPES.items():
            training = reader.read(ROOT / recipe["training"], recipe["sha256"])
            dataset = training["dataset_manifest"]
            reader.freeze(dataset["path"], DATASET_SHA)
            if dataset["sha256"] != DATASET_SHA:
                raise ValueError("Frozen 435 dataset identity changed")
            saved = {oid: _saved_outputs(oid, scorer, training, reader,
                {r["source_npz"]: r["source_npz_sha256"] for r in rows if r["object_id"] == oid})
                for oid in sorted({r["object_id"] for r in rows})}
            max_error, checked = 0., 0
            models[scorer] = {}
            for gesture in GESTURES:
                ref = training["internal_heads"][str(gesture)]
                reader.freeze(ref["path"], ref["sha256"])
                checkpoint = torch.load(ref["path"], map_location="cpu", weights_only=True)
                fit = set(checkpoint.get("fit_mesh_sha256", []))
                if (checkpoint.get("schema") != "anydex_boya_decision_state_dict_v1"
                        or checkpoint.get("stage") != "internal_check" or checkpoint.get("gesture_id") != gesture
                        or checkpoint.get("epoch") != 100 or checkpoint.get("training_variant") != recipe["variant"]
                        or checkpoint.get("dataset_manifest_sha256") != DATASET_SHA
                        or checkpoint.get("architecture_kwargs") != architecture
                        or checkpoint.get("training_config") != training["config"]
                        or not fit or not fit <= set(training["mesh_groups"]["train"])
                        or fit & set(training["mesh_groups"]["internal_check"])):
                    raise ValueError("Original fixed 32-fit/internal head contract violated")
                model = cls(**architecture)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()
                indices = [i for i, r in enumerate(rows) if r["gesture_id"] == gesture]
                with torch.inference_mode():
                    for start in range(0, len(indices), 128):
                        ids = indices[start:start + 128]
                        prediction, _ = model(torch.from_numpy(np.stack([features[i] for i in ids])))
                        forward_batches += 1
                        if tuple(prediction.shape) != (len(ids), 1, 20) or not torch.isfinite(prediction).all():
                            raise ValueError("Unexpected original head forward result")
                        for i, current in zip(ids, prediction[:, 0].numpy()):
                            row = rows[i]
                            key = (row["source_candidate_index"], gesture, row["raw_depth_index"], row["if_flip"])
                            old = saved[row["object_id"]].get(key)
                            if old is None:
                                raise ValueError("Candidate is missing from other recipe's saved queries")
                            error = float(np.max(np.abs(current.astype(float) - old.astype(float))))
                            if error > FORWARD_ATOL:
                                raise ValueError(f"Original saved head forward mismatch: {scorer}/{row['candidate_id']}: {error}")
                            max_error = max(max_error, error); checked += 1
                            row["scores"][scorer] = float(old[row["target_index"]])
                            row.setdefault("outputs20_identity", {})[scorer] = array_ref(old)
                            row.setdefault("cpu_forward_max_abs_error", {})[scorer] = error
                models[scorer][str(gesture)] = dict(path=ref["path"], sha256=ref["sha256"],
                    strict_load=True, stage="internal_check", epoch=100, final_relu_inference=True,
                    training_variant=recipe["variant"], cpu_scored_candidates=len(indices))
                del model
            audit[scorer] = dict(checked_candidates=checked, checked_outputs=20 * checked,
                                 max_abs_error=max_error, absolute_tolerance=FORWARD_ATOL)
        for row, origin in zip(rows, origins):
            directory = Path(row["generation_json"]).parent.parent
            freeze = reader.read(directory / "RANKING_FREEZE.json")
            recipe_sha = freeze["training_results_ref"]["sha256"]
            matches = [key for key, value in RECIPES.items() if value["sha256"] == recipe_sha]
            if len(matches) != 1 or origin["learned_score"] != row["scores"][matches[0]]:
                raise ValueError("Saved geometry admission score differs from original head output")
            row["generation_scorer"] = matches[0]
            row["generation_score_exactly_verified"] = True
    finally:
        torch.set_num_threads(threads)
    reader.verify()
    source_shas = {str(p.resolve()): reader.files[str(p.resolve())] for p in source_paths}
    model_shas = {m["path"]: m["sha256"] for family in models.values() for m in family.values()}
    result = dict(schema="anydex_boya_fixed_pool_scores_v1", catalog_sha256=json_sha(catalog),
        candidates=rows, objects=rank_candidates(rows), models=models,
        independent_cpu_forward_audit=audit, input_sha256=dict(sorted(reader.files.items())),
        source_sha256=dict(sorted(source_shas.items())), model_sha256=dict(sorted(model_shas.items())),
        catalog_field_whitelist=["candidate_id", "object_id", "generation_json", "generation_sha256",
            "candidate_dir", "record_index", "source_candidate_index", "gesture_id", "raw_depth_index",
            "extra_depth_index", "if_flip"],
        catalog_history_metadata_used_for_ranking=False,
        tie_break_fields=list(TIE_FIELDS), score_source="Existing frozen float32 head outputs, independently reloaded on CPU",
        strict_threshold=THRESHOLD, history_known=True, outcomes_read=False,
        new_head_forward_batches=forward_batches, new_cgr_forwards=0, optimizer_steps=0, physics_steps=0,
        boundary=BOUNDARY)
    json_sha(result)
    return result


def _top_metrics(ids, labels, k):
    chosen = ids[:k]
    known = [labels[cid] for cid in chosen if labels[cid] is not None]
    success = sum(known)
    unknown = len(chosen) - len(known)
    return dict(requested_k=k, effective_k=len(chosen), candidate_ids=chosen,
        success_count=success, unresolved_count=unknown,
        contains_success=True if success else (None if unknown else False),
        precision_at_k=(success / len(chosen) if chosen and not unknown else None),
        precision_bounds=([success / len(chosen), (success + unknown) / len(chosen)] if chosen else None))


def summarize(catalog, scores, outcomes):
    """Summarize inherited executor labels; infrastructure errors stay unknown.

    outcomes={'candidates':[{'candidate_id','object_id','status':'valid'|
    'infrastructure_error'|'pending','task_success':bool|null,'physical_steps':int,
    'physical_gates_passed':bool,'early_stop':bool}, ...]}. The driver validates
    actual original physical gates; this function never invents substitute gates.
    """
    catalog_index = {r["candidate_id"]: r for r in catalog_rows(catalog)}
    if scores.get("catalog_sha256") != json_sha(catalog):
        raise ValueError("Scores do not bind this frozen catalog")
    score_index = {r["candidate_id"]: r for r in scores["candidates"]}
    if len(score_index) != len(scores["candidates"]) or set(score_index) != set(catalog_index):
        raise ValueError("Score candidate set differs from frozen catalog")
    for cid, row in score_index.items():
        if row["object_id"] != catalog_index[cid]["object_id"]:
            raise ValueError("Score object identity mismatch")
    ranked = rank_candidates(scores["candidates"])
    if ranked != scores["objects"]:
        raise ValueError("Frozen ranking was changed")
    supplied = outcomes.get("candidates") if isinstance(outcomes, dict) else outcomes
    if not isinstance(supplied, list):
        raise ValueError("Explicit outcome list required")
    labels, states, seen = {cid: None for cid in catalog_index}, {}, set()
    for outcome in supplied:
        cid = outcome["candidate_id"]
        if cid in seen or cid not in catalog_index or outcome["object_id"] != catalog_index[cid]["object_id"]:
            raise ValueError("Outcome candidate identity duplicate or mismatch")
        seen.add(cid)
        state = outcome.get("status")
        if state not in ("valid", "infrastructure_error", "pending"):
            raise ValueError("Explicit valid/infrastructure_error/pending outcome status required")
        states[cid] = state
        if state == "valid":
            label = outcome.get("task_success")
            if (type(label) is not bool or type(outcome.get("physical_steps")) is not int
                    or outcome["physical_steps"] <= 0 or type(outcome.get("physical_gates_passed")) is not bool
                    or (label and (not outcome["physical_gates_passed"] or outcome.get("early_stop") is True))):
                raise ValueError("Valid outcome requires actual physics and inherited gate-consistent bool success")
            labels[cid] = int(label)
        elif outcome.get("task_success") is not None:
            raise ValueError("Infrastructure or pending outcome must not be labeled negative")
    objects = []
    for group in ranked:
        ids = group["rankings"][SCORERS[0]]
        n, success = len(ids), sum(labels[cid] == 1 for cid in ids)
        unresolved = sum(labels[cid] is None for cid in ids)
        item = dict(object_id=group["object_id"], candidate_count=n, valid_outcomes=n - unresolved,
            success_count=success, failure_count=n - unresolved - success, unresolved_count=unresolved,
            infrastructure_errors=sum(states.get(cid) == "infrastructure_error" for cid in ids),
            any_success_oracle=True if success else (None if unresolved else False),
            random_top1_success_expectation=success / n if not unresolved else None,
            random_top1_success_bounds=[success / n, (success + unresolved) / n], scorers={})
        for scorer, ordered in group["rankings"].items():
            top = {str(k): _top_metrics(ordered, labels, k) for k in (1, 3, 5)}
            admitted = [cid for cid in ordered if score_index[cid]["scores"][scorer] > THRESHOLD]
            selected = admitted[0] if admitted else None
            threshold = dict(value=THRESHOLD, comparison="strictly_greater_float32_0.85",
                admitted_count=len(admitted), abstain=not admitted, selected_candidate_id=selected,
                selected_success=(bool(labels[selected]) if selected and labels[selected] is not None else None),
                policy_success=(False if selected is None else (bool(labels[selected]) if labels[selected] is not None else None)),
                deployment_threshold_applicable=scorer != "cgr")
            item["scorers"][scorer] = dict(topk=top, strict_threshold=threshold)
        objects.append(item)
    aggregate = dict(object_count=len(objects), candidate_count=len(catalog_index),
        valid_outcomes=sum(r["valid_outcomes"] for r in objects),
        success_count=sum(r["success_count"] for r in objects),
        unresolved_count=sum(r["unresolved_count"] for r in objects), scorers={})
    for scorer in SCORERS:
        aggregate["scorers"][scorer] = {}
        for k in (1, 3, 5):
            metrics = [r["scorers"][scorer]["topk"][str(k)] for r in objects]
            covered = [r["contains_success"] for r in metrics]
            precisions = [r["precision_at_k"] for r in metrics]
            aggregate["scorers"][scorer]["top" + str(k)] = dict(
                objects_with_success=sum(v is True for v in covered),
                unresolved_objects=sum(v is None for v in covered), denominator=len(objects),
                contains_success_rate=(sum(covered) / len(covered) if None not in covered else None),
                macro_precision_at_k=(sum(precisions) / len(precisions) if None not in precisions else None))
    return dict(schema="anydex_boya_fixed_pool_summary_v1", catalog_sha256=json_sha(catalog),
        scores_sha256=json_sha(scores), outcomes_sha256=json_sha(outcomes),
        complete=aggregate["unresolved_count"] == 0, objects=objects, aggregate=aggregate,
        boundary=BOUNDARY, precision_denominator="min(k, object candidate count); unresolved trials remain unknown",
        early_physical_failures_included=True, infrastructure_errors_are_negative_labels=False)
