"""Independently verify saved augmentation/provenance and plot pad semantics.

Consumes artifacts only. No model load, optimizer, simulator or hand FK import.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_reference_geometry import angle, sha
from deploy_score_first import PROFILES, Reader, make_queries, validate_cached_forward
from run_augmented_representation import verify_files


def review(protocol_path, output):
    if output.exists():
        raise ValueError('Keep the existing review; choose a new directory')
    output.mkdir(parents=True)
    protocol = json.loads(protocol_path.read_text()); verify_files(protocol['frozen_files'])
    directory = Path(protocol['representation_dir'])
    receipt = json.loads((directory/'RESULTS.json').read_text())
    if receipt['status'] != 'complete' or receipt['network_forward_count'] != 11 or receipt['model_load_count'] != 1:
        raise ValueError('Eleven completed forwards from one model required')
    if receipt['protocol']['sha256'] != sha(protocol_path):
        raise ValueError('Receipt protocol mismatch')
    reader = Reader(); obs_path = Path(protocol['observation_json'])
    obs = reader.read(obs_path)
    validate_cached_forward(obs_path, obs, directory, receipt, reader)
    max_position_error = 0.; max_rotation_error = 0.; offset = 0
    with np.load(directory/'raw_camera_candidates.npz', allow_pickle=False) as merged:
        for index, branch in enumerate(receipt['branches']):
            for file in branch['output_files'].values():
                if sha(file['path']) != file['sha256']:
                    raise ValueError('Branch artifact changed')
            raw_path = Path(branch['output_files']['raw_camera_candidates.npz']['path'])
            recovered_path = Path(branch['output_files']['recovered_camera_candidates.npz']['path'])
            with np.load(raw_path, allow_pickle=False) as raw, np.load(recovered_path, allow_pickle=False) as recovered:
                count = len(raw['candidates']); end = offset+count
                # Independent homogeneous inverse, including the reflection.
                inverse = np.linalg.inv(np.asarray(branch['transform'], dtype=np.float64))
                expected_position = raw['candidates'][:, 12:]@inverse[:3, :3].T + inverse[:3, 3]
                expected_rotation = inverse[:3, :3]@raw['candidates'][:, 3:12].reshape(-1, 3, 3)
                if branch['scene_reflection']:
                    expected_rotation[:, :, 1] *= -1
                if count:
                    max_position_error = max(max_position_error, float(np.max(abs(expected_position-recovered['candidates'][:, 12:]))))
                    max_rotation_error = max(max_rotation_error, float(np.max(abs(expected_rotation-recovered['candidates'][:, 3:12].reshape(-1, 3, 3)))))
                for key in ('candidates', 'candidate_features'):
                    np.testing.assert_array_equal(merged[key][offset:end], recovered[key])
                np.testing.assert_array_equal(raw['candidate_features'], recovered['candidate_features'])
                np.testing.assert_array_equal(merged['source_branch'][offset:end], np.full(count, index))
                np.testing.assert_array_equal(merged['source_branch_row'][offset:end], np.arange(count))
                np.testing.assert_array_equal(merged['source_scene_reflection'][offset:end], np.full(count, branch['scene_reflection']))
                offset = end
        if max_position_error > 2e-7 or max_rotation_error > 5e-7 or offset != len(merged['candidates']):
            raise ValueError('Independent coordinate recovery audit failed')
        np.testing.assert_allclose(merged['candidates'][:, 2], merged['candidate_features'][:, -1], atol=1e-8, rtol=0)
        # Confirm the old score-first API can consume these arrays. This is a
        # pregeometry canonicalization diagnostic, NOT the deployed hand domain.
        queries, features, indices = make_queries(merged['candidates'], merged['candidate_features'], PROFILES['released-dh3-v1'], 'both')
        if len(queries) != 8*len(indices) or np.asarray(features).shape != (len(queries), 480):
            raise ValueError('Canonical query bridge failed')
        selected_branches = merged['source_branch'][indices]
        top_composition = {str(i):int(np.count_nonzero(selected_branches == i)) for i in range(11)}
        identity = {int(merged['source_branch_row'][i]) for i in indices if merged['source_branch'][i] == 0}
        identity_path = Path(protocol['identity_reference_dir'])/'raw_camera_candidates.npz'
        with np.load(identity_path, allow_pickle=False) as old:
            old_top = set(np.argsort(-old['candidates'][:, 0], kind='stable')[:1000].tolist())
        canonical = dict(cgr_limit=1000, thumb_flip_policy='both', queries=len(queries), feature_shape=[len(queries), 480],
                         global_CGR_top1000_branch_counts=top_composition,
                         old_identity_top1000_retained=len(identity & old_top),
                         scope='Generic pregeometry top1000 only; no Boya hand-domain/width/table/cloud admission, head score or physical feasibility claim')
    # The collision cloud is exactly the old identity sparse representation.
    root_cloud = directory/'representation_outputs.npz'
    if sha(root_cloud) != sha(Path(protocol['identity_reference_dir'])/'representation_outputs.npz'):
        raise ValueError('The original collision cloud changed')
    geometry_path = Path(protocol['geometry_output']); geometry = json.loads(geometry_path.read_text())
    if geometry['calibration']['sha256'] != sha(protocol['calibration']):
        raise ValueError('Geometry audit calibration changed')
    for path, record in geometry['source_files'].items():
        if sha(path) != record['sha256']: raise ValueError('Geometry source changed')
    extra_geometry = {}
    for g in (1, 6, 7, 8):
        rows = sorted((r for r in geometry['rows'] if r['gesture_id'] == g and r['opposing_group'] == 'ff'), key=lambda r:r['sample'])
        axes = [np.asarray(r['palm_T_reference'])[:3, 2] for r in rows]
        extra_geometry[str(g)] = dict(max_adjacent_approach_angle_deg=max(angle(a, b) for a,b in zip(axes[:-1], axes[1:])),
            opponent_normal_points_away_samples=sum(r['opponent_normal_dot_closing'] < 0 for r in rows),
            sample_count=len(rows), semantics='Static annotated normals only; no force-closure or penetration test')
    result = dict(schema='boya_reference_alignment_independent_review_v1', status='complete',
        protocol=dict(path=str(protocol_path), sha256=sha(protocol_path)),
        representation=dict(path=str(directory/'RESULTS.json'), sha256=sha(directory/'RESULTS.json')),
        geometry=dict(path=str(geometry_path), sha256=sha(geometry_path)),
        sources={str(Path(__file__).resolve()):sha(__file__)},
        branch_candidate_counts=[b['detail']['raw_candidates']['count'] for b in receipt['branches']],
        merged_rows=offset, all_paired_features_and_provenance_exact=True,
        independent_recovery_max_position_error_m=max_position_error,
        independent_recovery_max_rotation_element_error=max_rotation_error,
        identity_all_arrays_exact=receipt['identity_regression']['all_arrays_exactly_equal'],
        identity_sparse_collision_cloud_exact=True, canonical_bridge=canonical,
        geometry_extra=extra_geometry, frozen_files_unchanged=True,
        new_CGR_forwards=receipt['network_forward_count'], review_model_loads=0, new_physics_steps=0, new_training_steps=0,
        actual_inference_seconds=sum(b['detail']['timing_seconds']['full_representation_forward'] for b in receipt['branches']),
        runner_wall_seconds=receipt['timing_seconds']['total_wall'])
    # Static figures are exportable and link directly to the audited data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), constrained_layout=True)
    for i, g in enumerate((1, 6, 7, 8)):
        rows = sorted((r for r in geometry['rows'] if r['gesture_id'] == g and r['opposing_group'] == 'ff'), key=lambda r:r['sample'])
        fractions = np.arange(len(rows))/(len(rows)-1)
        axes[i, 0].plot(fractions, [1000*r['width_m'] for r in rows], label='thumb-index separation')
        axes[i, 0].plot(fractions, [1000*r['pair_midpoint_drift_to_closed_m'] for r in rows], label='midpoint distance to closed')
        axes[i, 0].set(ylabel=f'g{g}: distance (mm)', xlabel='closure fraction (0=open, 1=nominal closed)')
        start_axis = np.asarray(rows[0]['palm_T_reference'])[:3, 2]
        axes[i, 1].plot(fractions, [angle(start_axis, np.asarray(r['palm_T_reference'])[:3, 2]) for r in rows], label='approach change from open')
        if g != 6:
            axes[i, 1].plot(fractions, [r['first_two_opponents_plane_normal_angle_deg_mod180'] for r in rows], label='approach vs contact triangle (mod 180)')
        axes[i, 1].set(ylabel=f'g{g}: angle (degrees)', xlabel='closure fraction')
        for axis in axes[i]: axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.suptitle('Boya frozen pad references: static geometry hypotheses, no physical outcome')
    fig.savefig(output/'reference_geometry.png', dpi=160); plt.close(fig)
    (output/'AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Boya 参考实现对齐核验</title>
<style>body{max-width:1100px;margin:36px auto;font:17px/1.6 sans-serif}img{width:100%}code{background:#eee}</style>
<h1>Boya 参考实现对齐核验</h1><p>一个已知 A0 场景，原始点云＋10 个增强分支。仅离线验证；没有训练或物理试抓。</p>
<p>22,528 条候选保留分支及行号；identity 的全部数组与旧缓存完全一致。Boya 默认部署和旧结果不变。</p>
<p><a href="AUDIT.json">独立审计</a> · <a href="../PROTOCOL.json">冻结协议</a> · <a href="../representation/RESULTS.json">前向凭据</a> · <a href="../GEOMETRY_AUDIT.json">924 槽几何核验</a></p>
<p>图中为源指腹标注的几何量；接触点仍待人工审核。中点/方向随闭合变化，不等于物理失败已归因。</p>
<img src="reference_geometry.png" alt="Four hand types: separation, midpoint drift and approach-axis change through closure"></html>''')
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); review(args.protocol.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
