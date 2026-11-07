#!/usr/bin/env python3
"""Explicit fit-only BatchNorm-statistics control; never an automatic replacement.

No optimizer, labels or internal feature rows enter calibration. Original weights
are frozen; PyTorch update_bn receives exactly one tensor batch per hand type.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
sys.path.insert(0, str(ROOT / 'reproduction/boya'))
import train_boya_decision as original

SCHEMA = 'anydex_s40_fit_bn_reestimation_v1'
BASE_REPORT = ROOT / 'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json'
BASE_SHA256 = '447302d26c1e5cccc10f23ae897105a5d845abe661640cfc4ea5fa6d912ccac0'
UPDATE_BN_SHA256 = '953081be5ebce47dfe3ab5b1389cfa8150a82aaefa9c0e59bd50203e8a3910d8'
GESTURES = (1, 6, 7, 8)


def sha(path):
    return original.sha(path)


def helper_ref(expected=UPDATE_BN_SHA256):
    helper = inspect.unwrap(torch.optim.swa_utils.update_bn)
    digest = hashlib.sha256(inspect.getsource(helper).encode()).hexdigest()
    if digest != expected:
        raise ValueError('Original torch.optim.swa_utils.update_bn source changed')
    return dict(function='torch.optim.swa_utils.update_bn',
                path=str(Path(inspect.getsourcefile(helper)).resolve()),
                function_source_sha256=digest,
                file_sha256=sha(inspect.getsourcefile(helper)))


def fit_indices(metadata, checkpoint, gesture):
    ids = list(map(str, metadata['trial_id']))
    fit_ids = list(map(str, checkpoint['fit_trial_ids']))
    if len(ids) != len(set(ids)) or len(fit_ids) != len(set(fit_ids)) or len(fit_ids) < 2:
        raise ValueError('Unique source fit trial identities required')
    if any(metadata[k].shape != (len(ids),) for k in ('split', 'mesh_sha256', 'gesture_ids')):
        raise ValueError('Parallel metadata columns required')
    index = {tid: i for i, tid in enumerate(ids)}
    if any(tid not in index for tid in fit_ids):
        raise ValueError('Checkpoint fit trial missing from dataset')
    selected = np.asarray([index[tid] for tid in fit_ids], dtype=np.int64)
    if (set(selected) != set(np.flatnonzero(metadata['split'] == 'train'))
        or not (metadata['split'][selected] == 'train').all()
        or not (metadata['gesture_ids'][selected] == gesture).all()):
        raise ValueError('Calibration must contain exactly checkpoint train-only records')
    if set(map(str, metadata['mesh_sha256'][selected])) != set(checkpoint['fit_mesh_sha256']):
        raise ValueError('Checkpoint fit mesh identity mismatch')
    return selected


def read_fit_features(path, selected, row_count):
    """Read only selected .npy rows; no internal feature arrays are materialized."""
    with zipfile.ZipFile(path) as archive, archive.open('features.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Supported source .npy versions are 1 and 2')
        if shape != (row_count, 480) or fortran or dtype != np.dtype('float32'):
            raise ValueError('C-order source float32 canonical480 rows required')
        offset = stream.tell()
        rows = []
        for i in selected:
            stream.seek(offset + int(i) * 480 * dtype.itemsize)
            row = np.frombuffer(stream.read(480 * dtype.itemsize), dtype=dtype).copy()
            if row.shape != (480,) or not np.isfinite(row).all():
                raise ValueError('Invalid selected fit feature row')
            rows.append(row)
    return np.stack(rows)


def state_change_audit(model, before):
    after = model.state_dict()
    if set(after) != set(before):
        raise ValueError('State dictionary keys changed')
    allowed = set()
    for prefix, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            allowed.update((prefix + '.' if prefix else '') + suffix
                           for suffix in ('running_mean', 'running_var', 'num_batches_tracked'))
    changed = [k for k in before if not torch.equal(before[k], after[k].detach().cpu())]
    if set(changed) - allowed:
        raise ValueError('Non-BN-buffer state changed: ' + str(set(changed) - allowed))
    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter.detach().cpu()) or parameter.requires_grad:
            raise ValueError('Learnable parameter changed or was not frozen')
    return dict(all_parameters_equal=True, only_BN_running_buffers_may_change=True,
                allowed_changed_keys=sorted(allowed), changed_keys=changed,
                parameter_tensor_count=sum(1 for _ in model.parameters()))


def calibrate_model(model, features):
    if features.dtype != np.float32 or features.ndim != 2 or len(features) < 2:
        raise ValueError('At least two unique float32 fit rows required')
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.optim.swa_utils.update_bn([torch.from_numpy(features)], model, device=torch.device('cpu'))
    model.eval()
    audit = state_change_audit(model, before)
    return audit


def run(base_report, output, prepare_only=False):
    start = time.monotonic()
    base_report, output = Path(base_report).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Fresh output required; no overwrite or resume')
    if base_report != BASE_REPORT or sha(base_report) != BASE_SHA256:
        raise ValueError('Frozen source code100 report required')
    base = original.load(base_report)
    if base.get('status') != 'complete' or base.get('schema') != 'anydex_s40_code_recipe_training_v1':
        raise ValueError('Complete code100 source required')
    helper = helper_ref()
    dataset_ref = base['dataset_manifest']
    manifest_path = Path(dataset_ref['path'])
    if sha(manifest_path) != dataset_ref['sha256']:
        raise ValueError('Dataset manifest changed')
    manifest = original.load(manifest_path)
    inputs = {str(base_report): sha(base_report), str(manifest_path): sha(manifest_path)}
    source = {str(Path(__file__).resolve()): sha(__file__),
              str(Path(original.__file__).resolve()): sha(original.__file__),
              str(original.MODEL_SOURCE): sha(original.MODEL_SOURCE),
              helper['path']: helper['file_sha256']}
    for path, digest in base['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen base source changed: ' + path)
        source[path] = digest
    metadata, checkpoints, jobs = {}, {}, {}
    for gesture in GESTURES:
        g = str(gesture)
        ref = base['internal_heads'][g]
        checkpoint_path = Path(ref['path'])
        if sha(checkpoint_path) != ref['sha256']:
            raise ValueError('Source checkpoint changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if (checkpoint['stage'] != 'internal_check' or checkpoint['gesture_id'] != gesture
            or checkpoint['dataset_manifest_sha256'] != dataset_ref['sha256']):
            raise ValueError('Internal fit checkpoint identity mismatch')
        npz_ref = manifest['gesture_files'][g]
        npz = Path(npz_ref['path'])
        if not npz.is_absolute():
            npz = manifest_path.parent / npz
        if sha(npz) != npz_ref['sha256']:
            raise ValueError('Source NPZ changed')
        # Only metadata is accessed here, never labels or features.
        with np.load(npz, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in ('trial_id', 'split', 'mesh_sha256', 'gesture_ids')}
        selected = fit_indices(a, checkpoint, gesture)
        if any(mesh not in manifest['mesh_groups']['train'] for mesh in checkpoint['fit_mesh_sha256']):
            raise ValueError('Internal mesh crossed fit boundary')
        inputs[str(checkpoint_path)] = ref['sha256']
        inputs[str(npz.resolve())] = npz_ref['sha256']
        metadata[g], checkpoints[g] = a, checkpoint
        jobs[g] = dict(gesture_id=gesture, dataset_npz=str(npz.resolve()),
                       fit_trial_ids=checkpoint['fit_trial_ids'],
                       fit_mesh_sha256=checkpoint['fit_mesh_sha256'],
                       selected_indices=selected.tolist(), fit_records=len(selected),
                       internal_rows_excluded=int(np.sum(a['split'] == 'internal_check')),
                       source_checkpoint=dict(path=str(checkpoint_path), sha256=ref['sha256']))
    plan = dict(schema=SCHEMA, status='prepared',
                base_training_report_ref=dict(path=str(base_report), sha256=sha(base_report)),
                dataset_manifest=dataset_ref, config=base['config'], mesh_groups=base['mesh_groups'],
                input_sha256=inputs, source_sha256=source, helper=helper, jobs=jobs,
                calibration=dict(method=helper['function'], unique_full_fit_batches_per_head=1,
                    passes_per_head=1, labels_used=False, internal_features_used=False,
                    all_learnable_parameters_frozen=True, optimizer_steps=0, device='cpu'),
                model_selection='No model/epoch/threshold/calibration selection; explicit single control',
                new_physics_steps=0, new_CGR_forwards=0, EGAD_objects_accessed=0,
                deployment_default_changed=False)
    output.mkdir(parents=True)
    (output / 'internal_check').mkdir()
    original.save(output / 'PLAN.json', plan)
    report = dict(plan, plan_sha256=sha(output / 'PLAN.json'), internal_heads={}, heads={},
                  base_training=plan['base_training_report_ref'])
    original.save(output / 'RESULTS.json', report)
    if prepare_only:
        return report
    cls = original.official_components()[0]
    torch.set_num_threads(4)
    for gesture in GESTURES:
        g = str(gesture)
        checkpoint = checkpoints[g]
        model = cls(**checkpoint['architecture_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        job = jobs[g]
        features = read_fit_features(job['dataset_npz'], job['selected_indices'], len(metadata[g]['trial_id']))
        audit = calibrate_model(model, features)
        calibration = dict(plan_sha256=report['plan_sha256'], method=helper,
                           fit_trial_ids=job['fit_trial_ids'], fit_records=job['fit_records'],
                           feature_tensor_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                           batches=1, passes=1, labels_used=False, internal_features_used=False,
                           new_optimizer_steps=0, state_change_audit=audit,
                           base_checkpoint_ref=job['source_checkpoint'])
        saved = copy.deepcopy(checkpoint)
        saved['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        saved['post_training_calibration'] = calibration
        path = output / 'internal_check' / f'g{gesture:02d}.pt'
        torch.save(saved, path)
        reloaded = torch.load(path, map_location='cpu', weights_only=True)
        fresh = cls(**checkpoint['architecture_kwargs'])
        fresh.load_state_dict(reloaded['model_state_dict'], strict=True)
        fresh.requires_grad_(False)
        state_change_audit(fresh, checkpoint['model_state_dict'])
        row = dict(base['internal_heads'][g], path=str(path), sha256=sha(path),
                   base_head=base['internal_heads'][g], post_training_calibration=calibration,
                   strict_state_dict_reload=True)
        report['internal_heads'][g] = row
        original.save(output / 'RESULTS.json', report)
    # All checkpoints are immutable before diagnostic access to labels/internal features.
    freeze = dict(schema=SCHEMA, diagnostics_not_started=True,
                  heads={g: {k: row[k] for k in ('path', 'sha256')} for g, row in report['internal_heads'].items()})
    original.save(output / 'CALIBRATION_FREEZE.json', freeze)
    report['calibration_freeze_ref'] = dict(path=str(output / 'CALIBRATION_FREEZE.json'),
                                          sha256=sha(output / 'CALIBRATION_FREEZE.json'))
    for gesture in GESTURES:
        g = str(gesture)
        with np.load(jobs[g]['dataset_npz'], allow_pickle=False) as z:
            a = {k: z[k].copy() for k in ('features', 'labels', 'target_index', 'split')}
        checkpoint = torch.load(report['internal_heads'][g]['path'], map_location='cpu', weights_only=True)
        model = cls(**checkpoint['architecture_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        row = report['internal_heads'][g]
        row['fit_metric'] = original.evaluate(model, a, np.asarray(jobs[g]['selected_indices']), torch.device('cpu'))
        row['internal_check_metric'] = original.evaluate(model, a, np.flatnonzero(a['split'] == 'internal_check'), torch.device('cpu'))
    unchanged = all(Path(p).is_file() and sha(p) == h for p, h in {**inputs, **source}.items())
    frozen = all(sha(r['path']) == r['sha256'] for r in freeze['heads'].values())
    if not unchanged or not frozen or sha(output / 'PLAN.json') != report['plan_sha256']:
        raise ValueError('Frozen source/input/calibrated checkpoint changed')
    report.update(status='complete', inputs_and_sources_unchanged=unchanged,
                  calibrated_heads_unchanged_during_diagnostics=frozen,
                  diagnostics_used_for_selection=False, new_optimizer_steps=0,
                  BN_calibration_forward_batches=4, wall_s=time.monotonic() - start)
    for name in ('fit_metric', 'internal_check_metric'):
        total = sum(r[name]['count'] for r in report['internal_heads'].values())
        report[name + '_weighted'] = dict(count=total,
            mse=sum(r[name]['mse'] * r[name]['count'] for r in report['internal_heads'].values()) / total)
    original.save(output / 'RESULTS.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-training-report', type=Path, default=BASE_REPORT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    report = run(args.base_training_report, args.output_dir, args.prepare_only)
    print(json.dumps(dict(status=report['status'], output=str(args.output_dir),
                         internal_heads={g: {k: r[k] for k in ('path', 'sha256')}
                                         for g, r in report['internal_heads'].items()}), indent=2))


if __name__ == '__main__':
    main()
