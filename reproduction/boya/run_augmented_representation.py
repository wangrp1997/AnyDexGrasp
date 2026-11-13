"""Opt-in original identity + ten augmentations for ONE frozen Boya RGB-D.

Run with --protocol JSON. Original network/parser and native operators are reused;
DH3 rig gates, hand heads, geometry admission, control and physics are absent.
The protocol hashes source/assets/checkpoint and declares all preprocessing.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from augmented_candidates import augmentation_branches, merge_branches, recover_candidates
from run_pilot_representation import (PersistentOfficialCGR, ROOT, array_sha,
                                      numpy_rng_sha, read_frame, save, sha)


def verify_files(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Protocol source/input changed: '+path)


def capture_next_rng(engine):
    """Legacy forward restores its supplied state: supply the current stream.

    Capture AFTER the previous branch, rather than restoring the same initial
    state eleven times. This preserves the original sequential torch/NumPy RNG.
    """
    engine.cpu_rng = torch.get_rng_state().clone()
    engine.cuda_rng = torch.cuda.get_rng_state(engine.device).clone()
    engine.numpy_rng = copy.deepcopy(np.random.get_state())


def compare_arrays(directory, old_directory):
    result = {}
    for name in ('raw_camera_candidates.npz', 'representation_outputs.npz'):
        fields = {}
        with np.load(directory/name, allow_pickle=False) as new, np.load(old_directory/name, allow_pickle=False) as old:
            if set(new.files) != set(old.files):
                raise ValueError('Identity regression field names differ')
            for key in new.files:
                a, b = new[key], old[key]
                fields[key] = dict(shape=list(a.shape), array_sha256=array_sha(a),
                    exactly_equal=bool(a.dtype == b.dtype and np.array_equal(a, b)),
                    max_abs_error=float(np.max(abs(a.astype(float)-b.astype(float)))) if a.shape == b.shape and a.size else 0. if a.shape == b.shape else None)
        result[name] = fields
    return dict(all_arrays_exactly_equal=all(v['exactly_equal'] for f in result.values() for v in f.values()), files=result)


def run(protocol_path):
    protocol_digest = sha(protocol_path); protocol = json.loads(protocol_path.read_text())
    verify_files(protocol['frozen_files'])
    args = SimpleNamespace(**protocol['representation_args'])
    args.checkpoint, args.native_deps = Path(args.checkpoint), Path(args.native_deps)
    if (args.num_seed != 2048 or args.voxel_size != .005 or args.depth_min != .2 or args.depth_max != .65
            or protocol['branches'] != 11 or protocol['new_physics_steps'] != 0 or protocol['new_training_steps'] != 0):
        raise ValueError('This entry implements the declared original 11-branch Boya diagnostic only')
    directory = Path(protocol['representation_dir'])
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory/'RESULTS.json'
    start = time.monotonic(); engine = None
    report = dict(schema='boya_original_augmented_representation_v1', status='starting',
        protocol=dict(path=str(protocol_path), sha256=protocol_digest),
        scope='Known-scene offline identity + ten original augmentations; no physical or generalization claim',
        full_representation_forward=False, test_time_augmentation_branches=11,
        new_training_steps=0, new_physics_steps=0, hand_head_loaded=False,
        model_load_count=0, network_forward_count=0, branches=[], output_files={},
        excluded_filters=['DH3 camera approach/width/workspace gates'],
        downstream_policy='Boya score/geometry admission remains a separate explicit stage; no automatic promotion',
        timing_seconds={})
    save(result_path, report)
    try:
        row = dict(observation_json=protocol['observation_json'],
                   expected_observation_sha256=protocol['frozen_files'][protocol['observation_json']])
        points, _, binding, calibration, inputs, elapsed = read_frame(row, args)
        calibration['augmentation'] = 'Original identity then ten pre-generated original augment_data transforms; five scene reflections'
        report.update(scene_binding=binding, calibration=calibration, input_files=inputs,
                      input_points_sha256=array_sha(points))
        report['timing_seconds']['input_preparation'] = elapsed
        save(result_path, report)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable before model construction; zero model loads/forwards')
        begin = time.monotonic(); engine = PersistentOfficialCGR(args)
        report.update(model_load_count=1, checkpoint=engine.checkpoint, native_dependencies=engine.native,
                      representation_overlay=engine.overlay, candidate_parser_overlay=engine.parser_receipt)
        report['timing_seconds']['one_model_initialization'] = time.monotonic()-begin
        from np_utils import transform_point_cloud
        torch.set_rng_state(engine.cpu_rng); torch.cuda.set_rng_state(engine.cuda_rng, engine.device)
        np.random.set_state(engine.numpy_rng)
        transforms = augmentation_branches()  # original pre-generation BEFORE the identity forward
        report['augmentation_matrices'] = [m.tolist() for m, _ in transforms]
        report['rng_policy'] = dict(engine.rng, policy='One post-constructor stream, advanced across all 11 branches; all 10 transforms generated before identity',
                                   NumPy_after_augmentation_generation_sha256=numpy_rng_sha(np.random.get_state()))
        engine.rng = dict(report['rng_policy'])
        save(result_path, report)
        recovered = []
        for index, (matrix, scene_flip) in enumerate(transforms):
            branch_start = time.monotonic(); branch_dir = directory/'branches'/f'{index:02d}'
            branch_dir.mkdir(parents=True, exist_ok=False)
            transformed = transform_point_cloud(points, matrix).astype(np.float32)
            coords = np.ascontiguousarray(transformed/args.voxel_size, dtype=int)
            capture_next_rng(engine)
            before = dict(CPU=array_sha(engine.cpu_rng.numpy()), CUDA=array_sha(engine.cuda_rng.cpu().numpy()), NumPy=numpy_rng_sha(engine.numpy_rng))
            detail = engine.forward(transformed, coords, branch_dir)
            with np.load(branch_dir/'raw_camera_candidates.npz', allow_pickle=False) as raw:
                rows = recover_candidates(raw['candidates'], matrix, scene_flip, device=engine.device)
                features = raw['candidate_features'].copy()
            np.savez_compressed(branch_dir/'recovered_camera_candidates.npz', candidates=rows, candidate_features=features)
            recovered.append((rows, features))
            report['branches'].append(dict(index=index, scene_reflection=scene_flip,
                thumb_flip_applied=False, transform=matrix.tolist(), rng_before=before,
                input_points_sha256=array_sha(transformed), input_coords_sha256=array_sha(coords),
                output_files={p.name:dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(branch_dir.glob('*.npz'))},
                detail=detail, wall_seconds=time.monotonic()-branch_start))
            report['network_forward_count'] = engine.calls
            save(result_path, report)
            print(f'BOYA_AUGMENT branch={index} scene_reflection={scene_flip} rows={len(rows)}', flush=True)
        merged = merge_branches(recovered)
        merged['source_scene_reflection'] = np.array([f for _, f in transforms], dtype=bool)[merged['source_branch']]
        np.savez_compressed(directory/'raw_camera_candidates.npz', **merged)
        # Collision admission needs the ORIGINAL scene cloud, not reflected or
        # concatenated sparse inputs. Keep exactly the identity representation.
        shutil.copyfile(directory/'branches/00/representation_outputs.npz', directory/'representation_outputs.npz')
        report['identity_regression'] = compare_arrays(directory/'branches/00', Path(protocol['identity_reference_dir']))
        if protocol['require_identity_exact'] and not report['identity_regression']['all_arrays_exactly_equal']:
            raise ValueError('Identity no longer matches the frozen prior observation forward')
        for name in ('raw_camera_candidates.npz', 'representation_outputs.npz'):
            path = directory/name
            report['output_files'][name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        report.update(status='complete', full_representation_forward=True,
            raw_candidates=dict(count=len(merged['candidates']), feature_shape=list(merged['candidate_features'].shape),
                                coordinate_frame='original camera optical frame', deduplication=False,
                                order='identity then original branch order; original parser row order within branch'))
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    report['network_forward_count'] = engine.calls if engine else 0
    try:
        verify_files(protocol['frozen_files'])
        if sha(protocol_path) != protocol_digest:
            raise ValueError('Protocol changed during run')
        report['frozen_files_unchanged_at_end'] = True
    except Exception as exc:
        report.update(status='failed', frozen_files_unchanged_at_end=False, freeze_error=str(exc))
    report['timing_seconds']['total_wall'] = time.monotonic()-start
    save(result_path, report)
    print(json.dumps({k:report[k] for k in ('status', 'model_load_count', 'network_forward_count', 'timing_seconds')}, indent=2), flush=True)
    return report['status'] == 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    if not run(args.protocol.resolve()):
        sys.exit(1)


if __name__ == '__main__':
    main()
