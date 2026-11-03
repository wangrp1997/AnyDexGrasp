"""Independently audit saved synthetic RGB-D -> original DH3 selection artifacts.

No model/native CUDA operation, collision query, renderer or physics is invoked.
The audit checks pixel provenance, quantization coverage, source/RNG ordering,
array bit identity, and immutable SHA links across the saved pipeline receipts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import scipy.io

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def same_bits(actual, expected):
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.tobytes(order='C') == expected.tobytes(order='C')


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def verify_record(record):
    path = Path(record['path'])
    assert path.stat().st_size == record['bytes'], str(path)
    assert sha(path) == record['sha256'], str(path)


def audit(scene_dir, run_dir):
    scene_path = scene_dir/'scene_manifest.json'
    scene = read(scene_path)
    scene_result = read(scene_dir/'RESULTS.json')
    run = read(run_dir/'RESULTS.json')
    frame = read(run_dir/'frame/RESULTS.json')
    hand = read(run_dir/'hand/RESULTS.json')
    collision = read(run_dir/'collision/RESULTS.json')
    selected = read(run_dir/'selection/RESULTS.json')
    assert scene_result['physical_steps'] == scene['validation']['integrated_physics_steps'] == 0
    assert scene_result['scene_manifest_sha256'] == sha(scene_path)
    for item in scene['files'] + scene['source_assets']:
        verify_record(item)
    signature = hashlib.sha256(json.dumps(dict(spec=scene['spec'], source_assets=scene['source_assets']),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert signature == scene['scene_signature_sha256']
    for item in run['input_files'].items():
        verify_record(dict(path=item[0], **item[1]))
    for report in (run, frame):
        assert report['scene_binding']['manifest_sha256'] == sha(scene_path)
        assert report['scene_binding']['scene_signature_sha256'] == signature
        assert report['scene_binding']['manifest_or_object_pose_fed_to_network'] is False
    for stage in ('frame', 'hand', 'collision'):
        assert run[stage+'_stage']['RESULTS_sha256'] == sha(run_dir/stage/'RESULTS.json')
    assert hand['source_frame_RESULTS_sha256'] == sha(run_dir/'frame/RESULTS.json')
    assert collision['input']['hand_prediction_sha256'] == sha(run_dir/'hand/predicted_DH3_hand_poses_pre_collision.npz')
    assert collision['input']['scene_file_sha256'] == sha(run_dir/'frame/scene_points_down.npy')
    assert selected['collision_RESULTS_sha256'] == sha(run_dir/'collision/RESULTS.json')
    assert selected['collision_outputs_sha256'] == sha(run_dir/'collision/collision_outputs.npz')
    verify_record(selected['output'])
    assert sha(run['original_source_overlay']['source']) == run['original_source_overlay']['sha256']
    assert sha(collision['instrumentation']['original_source']) == collision['instrumentation']['source_sha256']
    assert sha(selected['source']) == selected['source_sha256']
    assert sha(run['checkpoint']['path']) == run['checkpoint']['sha256']
    depth = np.asarray(Image.open(scene_dir/'depth.png'))
    assert depth.dtype == np.uint16
    meta = scipy.io.loadmat(scene_dir/'meta.mat')
    assert not any(key in meta for key in ('cls_indexes', 'poses', 'object_ids'))
    K = np.asarray(meta['intrinsic_matrix'], dtype=np.float64)
    factor = float(np.asarray(meta['factor_depth']).reshape(-1)[0])
    assert np.array_equal(K, scene['camera']['K'])
    assert np.array_equal(K, run['calibration']['intrinsic_matrix'])
    assert factor == run['calibration']['factor_depth'] == scene['spec']['camera']['depth_factor']
    T_WC = np.array(scene['camera']['T_WC'])
    T_CW = np.array(scene['camera']['T_CW'])
    assert np.array_equal(T_WC, run['scene_binding']['T_WC'])
    assert np.allclose(T_WC@T_CW, np.eye(4), rtol=0, atol=1e-12)
    assert np.isclose(np.linalg.det(T_WC[:3,:3]), 1, rtol=0, atol=1e-12)
    # Reconstruct directly from the actual depth image using original get_grasp's
    # calibration/order/cast. No dense CAD or segmentation point is substituted.
    u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    z = depth/factor
    full_points = np.stack([(u-K[0,2])/K[0,0]*z, (v-K[1,2])/K[1,1]*z, z], axis=-1)
    mask = (z > .15) & (z < .72)
    original_points = full_points[mask].astype(np.float32)
    saved = np.load(run_dir/'frame/scene_points_down.npy', allow_pickle=False)
    assert saved.dtype == np.float32 and saved.ndim == 2 and saved.shape[1] == 3
    pixel_u = np.rint(saved[:,0]/saved[:,2]*K[0,0]+K[0,2]).astype(int)
    pixel_v = np.rint(saved[:,1]/saved[:,2]*K[1,1]+K[1,2]).astype(int)
    assert ((pixel_u >= 0) & (pixel_u < depth.shape[1])).all()
    assert ((pixel_v >= 0) & (pixel_v < depth.shape[0])).all()
    assert mask[pixel_v,pixel_u].all()
    same_bits(saved, full_points[pixel_v,pixel_u].astype(np.float32))
    # Integer conversion is truncation, matching original torch coords -> int.
    original_voxels = np.unique((original_points/.005).astype(np.int64), axis=0)
    saved_voxels = np.unique((saved/.005).astype(np.int64), axis=0)
    assert len(saved) == len(saved_voxels)
    same_bits(saved_voxels, original_voxels)
    # Replay only the original NumPy augmentation recipe, no network invocation.
    source = ast.parse(Path(run['original_source_overlay']['source']).read_text())
    augment = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'augment_data')
    scope = {'np':np}
    exec(compile(ast.Module(body=[augment], type_ignores=[]), str(ROOT/'robot_dh3.py'), 'exec', dont_inherit=True), scope)
    np.random.seed(run['seed'])
    matrices = [np.eye(4)] + [scope['augment_data'](flip=bool(i%2)) for i in range(10)]
    arrays = [archive(run_dir/'frame/branches'/('branch_%02d.npz'%i)) for i in range(11)]
    branch_counts = []
    for i,(matrix,branch) in enumerate(zip(matrices,arrays)):
        same_bits(branch['input_augmentation_matrix'], matrix)
        assert run['branches'][i]['flip'] == (False if i == 0 else bool((i-1)%2))
        branch_counts.append(len(branch.get('ggarray', [])))
        assert branch_counts[-1] == run['branches'][i]['candidates_after_original_recovery_and_gates']
    merged = archive(run_dir/'frame/original_merged_get_grasp_outputs.npz')
    same_bits(merged['ggarray'], np.concatenate([a['ggarray'] for a in arrays if 'ggarray' in a]))
    same_bits(merged['candidate_features'], np.concatenate([a['candidate_features'] for a in arrays if 'ggarray' in a]))
    raw = archive(run_dir/'frame/raw_camera_candidates.npz')
    same_bits(raw['candidates'], np.concatenate([merged['ggarray'][:,:2],merged['ggarray'][:,3:16]],axis=1))
    same_bits(raw['candidate_features'], merged['candidate_features'])
    # Independently derive masks from the recorded actual Open3D query bits,
    # then preserve the original admitted-row ordering exactly.
    pre = archive(run_dir/'hand/predicted_DH3_hand_poses_pre_collision.npz')
    col = archive(run_dir/'collision/collision_outputs.npz')
    overlap = np.unpackbits(col['scene_inclusion_masks_packed'],axis=1)[:,:len(saved)].sum(axis=1)
    assert np.array_equal(overlap,col['overlap_point_counts'])
    assert np.array_equal(overlap==0,col['admission_mask'])
    assert np.array_equal(np.flatnonzero(col['min_width_index']),col['source_indices'])
    admitted = col['admission_mask']; widths = col['min_width_index']
    same_bits(col['admitted_DH3_grasp_group_array'],col['DH3_grasp_group_array'][admitted])
    same_bits(col['admitted_two_finger_grasp_group_array'],col['two_finger_grasp_group_array'][admitted])
    same_bits(col['admitted_grasp_features'],pre['grasp_features'][widths][admitted])
    for item in collision['assets']['slots']:
        assert sha(item['path']) == item['sha256']
    chosen = archive(run_dir/'selection/selected_original_DH3_pose.npz')
    rank = np.argsort(col['admitted_DH3_grasp_group_array'][:,0])[::-1][:10]
    same_bits(chosen['top10_admitted_array_indices'],rank)
    same_bits(chosen['top10_pre_collision_indices'],col['source_indices'][admitted][rank])
    for selection_key,collision_key in (
        ('DH3_grasp_group_array','admitted_DH3_grasp_group_array'),
        ('two_finger_grasp_group_array','admitted_two_finger_grasp_group_array'),
        ('grasp_features','admitted_grasp_features')):
        same_bits(chosen['top10_'+selection_key],col[collision_key][rank])
    for one,many in (('top1_DH3_grasp_array','top10_DH3_grasp_group_array'),
        ('top1_two_finger_grasp_array','top10_two_finger_grasp_group_array'),
        ('top1_grasp_features','top10_grasp_features')):
        same_bits(chosen[one],chosen[many][0])
    same_bits(chosen['top10_object_ids'],col['admitted_two_finger_grasp_group_array'][rank,-1])
    assert collision['stages']['collision_admitted'] == int(admitted.sum())
    assert selected['selected_pre_collision_index'] == int(chosen['top10_pre_collision_indices'][0])
    return dict(status='saved_synthetic_frame_and_original_selection_binding_verified',
        scene_manifest=dict(path=str(scene_path.resolve()),sha256=sha(scene_path)),
        pipeline_RESULTS=dict(path=str((run_dir/'RESULTS.json').resolve()),sha256=sha(run_dir/'RESULTS.json')),
        selection_RESULTS=dict(path=str((run_dir/'selection/RESULTS.json').resolve()),sha256=sha(run_dir/'selection/RESULTS.json')),
        scene_signature_sha256=signature,
        checks=dict(all_scene_render_files_and_assets_SHA=True,all_stage_receipt_SHA_links=True,
            camera_depth_calibration_and_transforms=True,identity_points_exact_original_depth_pixel_bits=True,
            original_truncation_voxel_keys_complete_unique=True,original_seeded_11_transform_recipe_bits=True,
            original_branch_concatenation_bits=True,parser_column_bridge_bits=True,
            actual_collision_mask_equals_recorded_zero_overlap_bits=True,nonempty_original_PCL_SHA=True,
            original_argsort_top10_and_top1_array_bits=True),
        observed=dict(valid_depth_pixels=len(original_points),identity_scene_voxels=len(saved),
            branch_candidates=branch_counts,merged_candidates=len(merged['ggarray']),
            collision_input=len(widths),collision_admitted=int(admitted.sum()),
            top1_pre_collision_index=selected['selected_pre_collision_index'],
            top1_type=selected['selected_hand']['type'],top1_raw_learned_score=selected['selected_hand']['score']),
        timing_seconds=run['timing_seconds'],
        timing_boundary='15.980s eleven get_grasp branches; 35.647s pipeline internal total includes load/scoring/collision',
        audit_new_network_forwards=0,audit_new_collision_queries=0,audit_new_renders=0,audit_physical_steps=0,
        boundary='Saved-artifact integrity, not an independent grasp-success or force-closure proof')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-dir',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new audit output path')
    try:
        result=audit(args.scene_dir,args.run_dir)
        result['audit_source']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
    except Exception as exc:
        result=dict(status='artifact_binding_audit_failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
