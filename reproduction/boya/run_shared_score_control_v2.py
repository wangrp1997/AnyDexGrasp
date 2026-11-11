#!/usr/bin/env python3
"""Explicit serialization-only recovery of the bounded shared-score control.

V1 completed its first1000 updates, then failed safe checkpoint reload because
fit mesh IDs were NumPy string scalars. This entry converts metadata to Python
scalars when saving; tensors, training, folds, metrics and acceptance are kept.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import run_shared_score_control as base

FAILED = base.ROOT / "outputs/boya_shared_score_20261111_v1"
SCHEMA = "anydex_shared_score_serialization_recovery_v2"


def native_metadata(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {native_metadata(k): native_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native_metadata(v) for v in value]
    if isinstance(value, tuple):
        return tuple(native_metadata(v) for v in value)
    return value


def recovery_contract():
    return {
        "schema": SCHEMA,
        "failed_run": str(FAILED),
        "failed_protocol_sha256": base.sha(FAILED / "PROTOCOL.json"),
        "failed_progress_sha256": base.sha(FAILED / "PROGRESS.json"),
        "failed_checkpoint_sha256": base.sha(FAILED / "cv_fold0/condition_linear_seed20261111.pt"),
        "change": "At torch.save only, recursively convert numpy scalar metadata to Python scalars; leave every tensor untouched. Record completed fit history before reload.",
        "old_failed_actual_updates": 1000,
        "old_progress_counter_erratum": "V1 counter advanced only after strict reload, hence reported0 although first fit completed1000 updates before torch.save. Original progress/log retained.",
        "fresh_run_updates": 14500,
        "total_including_failed_attempt": 15500,
        "parameters_folds_seed_epochs_loss_and_gate_changed": False,
        "new_CGR_forwards": 0, "new_physics_steps": 0,
    }


def execute(output, stage):
    original_freeze = base.freeze_sources
    original_save = torch.save
    original_fit = base.fit_model
    journal = []

    def sources():
        result = original_freeze()
        result[str(Path(__file__).resolve())] = base.sha(__file__)
        return result

    def safe_save(value, *args, **kwargs):
        return original_save(native_metadata(value), *args, **kwargs)

    def measured_fit(x, labels, object_ids, family, seed):
        model, history = original_fit(x, labels, object_ids, family, seed)
        journal.append({"fit_number": len(journal)+1, "family": family, "seed": seed,
                        "records": len(labels), "optimizer_steps": len(history),
                        "completed_utc": datetime.now(timezone.utc).isoformat(), "history": history})
        base.save(output / "FIT_COMPLETIONS.json", {"schema": SCHEMA, "fits": journal,
                    "optimizer_steps": sum(r["optimizer_steps"] for r in journal)})
        return model, history

    base.freeze_sources = sources
    torch.save = safe_save
    base.fit_model = measured_fit
    try:
        if stage == "prepare":
            failed = base.read(FAILED / "PROGRESS.json")
            if failed["status"] != "failed_no_implicit_resume" or "numpy._core.multiarray.scalar" not in failed["error"]:
                raise ValueError("Expected documented V1 serialization failure")
            base.verify_hashes(base.read(FAILED / "PROTOCOL.json")["source_sha256"])
            protocol = base.prepare(output)
            protocol["serialization_recovery"] = recovery_contract()
            base.save(output / "PROTOCOL.json", protocol)
            base.save(output / "RECOVERY_DECLARATION.json", protocol["serialization_recovery"])
            return protocol
        protocol = base.read(output / "PROTOCOL.json")
        if protocol.get("serialization_recovery") != recovery_contract():
            raise ValueError("Recovery declaration changed")
        return base.run(output)
    finally:
        base.freeze_sources = original_freeze
        torch.save = original_save
        base.fit_model = original_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["prepare", "run"])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(base.ROOT / "outputs") or output == FAILED:
        raise ValueError("A separate output under AnyDex outputs is required")
    execute(output, args.stage)


if __name__ == "__main__":
    main()
