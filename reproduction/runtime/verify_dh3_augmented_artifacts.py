"""Audit saved original eleven-branch outputs without GPU/model/collision reruns.

Pre-recovery parser arrays were not persisted; this script does not pretend to
numerically replay that missing boundary. It verifies original-source binding,
seeded transforms, recovered output gates/rotations, exact merge and handoff.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from run_official_dh3_collision import sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--identity-frame',type=Path)
    args=parser.parse_args();root=args.run_dir
    receipt=json.loads((root/'RESULTS.json').read_text())
    assert receipt['status']=='eleven_branch_original_offline_pipeline_complete'
    assert receipt['test_time_augmentation_branches']==11 and receipt['checkpoint']['model_load_count']==1
    assert receipt['native_operator_call_counts']=={'furthest_point_sample':11,'cylinder_query':11,'grouping_operation':22}
    for key in ['frame_stage','hand_stage','collision_stage']:
        stage=receipt[key];assert sha(Path(stage['directory'])/'RESULTS.json')==stage['RESULTS_sha256']
    source=receipt['original_source_overlay']
    assert sha(source['source'])==source['sha256']
    assert source['original_recovery_flip_filters_and_aggregation_unchanged']
    assert set(sum([x['names'] for x in source['camera_constant_changes']],[]))=={'fx','fy','cx','cy','s'}
    rng=np.random.RandomState(receipt['seed']);matrices=[np.eye(4)]
    for index in range(10):
        angle=rng.random_sample()*np.pi/3-np.pi/6
        c,s=np.cos(angle),np.sin(angle)
        x,y=rng.random_sample(2)*.1-.05
        reflection=np.diag([-1,1,1,1]) if index%2 else np.eye(4)
        rotation=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        translation=np.eye(4);translation[0,3]=x;translation[1,3]=y
        matrices.append((translation@(rotation@reflection).astype(np.float32)).astype(np.float32))
    grasps=[];features=[];branch_counts=[];rot_errors=[];branch_sha={}
    for index,(entry,matrix) in enumerate(zip(receipt['branches'],matrices)):
        path=root/'frame/branches'/('branch_%02d.npz'%index)
        branch_sha[path.name]=sha(path)
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(matrix,saved['input_augmentation_matrix'])
            assert np.array_equal(matrix,np.asarray(entry['input_augmentation_matrix']))
            expected_post=matrix.copy()
            if entry['flip']:expected_post[:,0]*=-1
            assert np.array_equal(expected_post,np.asarray(entry['post_original_flip_mutation_matrix']))
            assert entry['flip']==bool(index and (index-1)%2)
            assert entry['original_parser_candidates_before_recovery_and_gates']==2048
            gg=saved['ggarray'];feat=saved['candidate_features']
            assert len(gg)==len(feat)==entry['candidates_after_original_recovery_and_gates']
            assert np.isfinite(gg).all() and np.isfinite(feat).all()
            assert np.all((gg[:,10]>.9)&(gg[:,1]<.099)&(gg[:,1]>.045))
            assert np.all((gg[:,13]>-.25)&(gg[:,13]<.25)&(gg[:,14]>-.205)&(gg[:,14]<.03))
            assert np.array_equal(gg[:,3],feat[:,-1])
            rotations=gg[:,4:13].reshape(-1,3,3)
            err=max(np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3))),np.max(np.abs(np.linalg.det(rotations)-1)))
            assert err<1e-5
            rot_errors.append(float(err));grasps.append(gg.copy());features.append(feat.copy());branch_counts.append(len(gg))
    merged=np.concatenate(grasps);merged_features=np.concatenate(features)
    with np.load(root/'frame/original_merged_get_grasp_outputs.npz',allow_pickle=False) as saved:
        assert np.array_equal(merged,saved['ggarray']) and np.array_equal(merged_features,saved['candidate_features'])
    with np.load(root/'frame/raw_camera_candidates.npz',allow_pickle=False) as saved:
        assert np.array_equal(np.concatenate([merged[:,:2],merged[:,3:16]],axis=1),saved['candidates'])
        assert np.array_equal(merged_features,saved['candidate_features'])
    # Independently reproduce the original sign canonicalization and global top1000 handoff.
    rots=merged[:,4:13].reshape(-1,3,3).copy();flip=rots[:,1,1]>0
    rots[flip,:,1:3]*=-1
    canonical=merged.copy();canonical[:,4:13]=rots.reshape(-1,9)
    canonical_features=np.c_[merged_features,flip]
    order=np.argsort(merged[:,0])[::-1][:1000]
    ordered_features=canonical_features[order].copy()
    canonical_cgr=[]
    for row in ordered_features:
        scores=row[240:480].reshape(48,5).copy()
        widths=row[480:720].reshape(48,5).copy()
        if row[-1]:
            scores=np.roll(scores,24,axis=0);widths=np.roll(widths,24,axis=0)
        # Original feature helper mutates half-swapped input views in place,
        # then returns a separately angle-rolled 480-vector to the classifier.
        row[240:480]=scores.ravel();row[480:720]=widths.ravel()
        angle=int(row[-3]+.1)
        canonical_cgr.append(np.r_[np.roll(scores,-angle,axis=0).ravel(),np.roll(widths,-angle,axis=0).ravel()])
    with np.load(root/'hand/predecision_inputs.npz',allow_pickle=False) as saved:
        assert np.array_equal(canonical[order],saved['ggarray'])
        assert np.array_equal(ordered_features,saved['grasp_features'])
        assert np.array_equal(np.array(canonical_cgr),saved['canonical_CGR'])
    assert len(merged)==receipt['merged_candidates_after_original_get_grasp_gates']
    identity_equal=None
    if args.identity_frame:
        with np.load(args.identity_frame/'representation_outputs.npz',allow_pickle=False) as saved:
            identity_equal=np.array_equal(saved['sparse_input_XYZ_features'],np.load(root/'frame/scene_points_down.npy',allow_pickle=False))
        assert identity_equal
    mapping=json.loads((root/'hand/INDEPENDENT_MAPPING_VERIFICATION.json').read_text())
    collision=json.loads((root/'collision/INDEPENDENT_ARTIFACT_VERIFICATION.json').read_text())
    selection=json.loads((root/'selection/RESULTS.json').read_text())
    assert mapping['all_checks_passed'] and collision['all_passed'] and selection['selected']
    result={'all_checks_passed':True,'scope':'Saved eleven-branch software artifacts, no model/collision rerun',
        'seeded_augmentation_matrices_exact':True,'original_flip_mutation_matrices_exact':True,
        'branches':11,'branch_candidate_counts':branch_counts,'merged_candidates':len(merged),
        'branch_concatenation_and_parser_column_handoff_exact':True,
        'all_recovered_outputs_pass_original_camera_width_workspace_gates':True,
        'rotation_orthogonality_and_det_max_error':max(rot_errors),
        'independent_flip_canonicalization_and_top1000_exact':True,
        'independent_inplace_CGR_half_swap_and_angle_roll_exact':True,
        'identity_collision_scene_matches_prior_identity_frame':identity_equal,
        'mapping':mapping,'collision':collision,'selected_hand':selection['selected_hand'],
        'pre_recovery_boundary':'Each branch saved post-recovery/gated arrays only. Actual raw 2048 parser arrays were not persisted, so numerical pre/post recovery replay cannot be independently audited from artifacts. Original recovery function is bound by unchanged author source SHA, and saved recovered outputs, gates and merge are audited.',
        'physical_grasps_executed':0,'grasp_success_claim':False,
        'source_RESULTS_sha256':sha(root/'RESULTS.json'),'branch_artifact_sha256':branch_sha}
    (root/'INDEPENDENT_PIPELINE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['all_checks_passed','branch_candidate_counts','merged_candidates','rotation_orthogonality_and_det_max_error','identity_collision_scene_matches_prior_identity_frame']},indent=2))


if __name__=='__main__':main()
