"""Verify original DH3 lookup transforms and ranked metadata independently."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from validate_official_decision import sha


def transform(rotation,translation):
    matrix=np.eye(4)
    matrix[:3,:3]=rotation;matrix[:3,3]=translation
    return matrix


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    receipt=json.loads((args.run_dir/'RESULTS.json').read_text())
    assert receipt['RANDOM'] is False and not receipt['collision_check_executed']
    with np.load(args.run_dir/'predicted_DH3_hand_poses_pre_collision.npz',allow_pickle=False) as saved:
        hand=saved['DH3_grasp_group_array'];grasp=saved['two_finger_grasp_group_array']
    table_path=Path(receipt['hand_mapping']['table']);table=json.loads(table_path.read_text())
    errors=[];angle_errors=[];rot_errors=[];keys=[]
    for h,g in zip(hand,grasp):
        key=str(np.round(h[18]*100,1));slot=table['pose'+str(int(h[2]))][key]
        original_grasp=transform(g[4:13].reshape(3,3),g[13:16])
        hand_pose=transform(h[3:12].reshape(3,3),h[12:15])
        calibration=transform(np.asarray(slot['rotation']),np.asarray(slot['translation']))
        errors.append(float(np.max(np.abs(hand_pose@calibration-original_grasp))))
        angle_errors.append(float(np.max(np.abs(h[15:17]-slot['2d']))))
        rot_errors.append(float(np.max(np.abs(hand_pose[:3,:3].T@hand_pose[:3,:3]-np.eye(3)))))
        keys.append('pose'+str(int(h[2]))+'/'+key)
    assert max(errors)<1e-6 and max(angle_errors)==0 and max(rot_errors)<1e-5
    type_widths={1:(0.,.099),2:(.007,.09),3:(0.,.106),4:(.006,.063)}
    expected_widths=np.array([min(min(type_widths[int(t)][1],.099),max(max(type_widths[int(t)][0],.01),float(width))) for t,width in zip(hand[:,2],grasp[:,1])])
    assert np.array_equal(hand[:,18],expected_widths)
    width_representation_error=float(np.max(np.abs(hand[:,18]-grasp[:,1])))
    additional=hand[:,1]-grasp[:,3]
    depth_grid=np.array([0.,.01,.02,.03])
    assert np.min(np.abs(additional[:,None]-depth_grid),axis=1).max()<1e-7
    with np.load(args.run_dir/'learned_ranked_proposals.npz',allow_pickle=False) as ranked:
        scores=ranked['scores'];mask=ranked['score_gate_mask']
        assert np.array_equal(mask,scores>.85)
        index=np.argsort(scores[mask])[::-1][:400]
        assert np.array_equal(hand[:,0],scores[mask][index])
        assert np.array_equal(hand[:,2],ranked['types'][mask][index])
    report={'scope':'independent predicted-hand lookup/score consistency only; no collision admission or grasp execution',
        'all_checks_passed':True,'predicted_hand_poses':len(hand),
        'T_hand_times_T_calibration_equals_T_twofinger_max_error':max(errors),
        'actuator_2d_angle_lookup_max_error':max(angle_errors),
        'hand_rotation_orthogonality_max_error':max(rot_errors),
        'hand_width_equals_independent_original_clamp_exactly':True,
        'source_float32_vs_hand_float64_width_max_difference_m':width_representation_error,
        'additional_depth_on_original_0_1_2_3cm_grid':True,
        'original_point85_score_mask_preserved':True,'original_top400_score_and_type_order_exact':True,
        'unique_calibration_slots':len(set(keys)),
        'collision_check_executed':False,'physical_grasps_executed':0,
        'artifact_sha256':{name:sha(args.run_dir/name) for name in ['RESULTS.json','predecision_inputs.npz','learned_ranked_proposals.npz','predicted_DH3_hand_poses_pre_collision.npz']}}
    (args.run_dir/'INDEPENDENT_MAPPING_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
