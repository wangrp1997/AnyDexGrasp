"""Prepare Boya geometry and validate its interface to the existing complete arm.

This is a zero-physics interface check. Replayed published CGR candidates have
already passed the DH3 demo gates; they are NOT a new Boya training collection.
The saved requests remain explicitly unadmitted until full scene/path checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from asset_adapter import BoyaAssets, DEFAULT_CANONDEX, DEFAULT_URDF, DEFAULT_SCENE, map_slot, rigid, sha

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check_full_arm(assets, calibration, config):
    from canondex.planning.model import PlanningModel
    from canondex.grasp.template_arm_bridge import with_hand_command
    model = PlanningModel(Path(config['robot_urdf']))
    # This retained home is a configuration reference only. No old successful
    # grasp is selected, replayed or relabelled as an AnyDex grasp.
    plan = json.loads(Path(config['reference_plan']).read_text())
    initial = model.full_state(plan['initial'])
    worst = 0.
    for state in calibration['states']:
        family = assets.families[state['gesture_id']]
        q = np.asarray(state['independent_q'])
        seed = initial | family.wrist
        full = with_hand_command(model, seed, assets.hand.names, q)
        frames = model.fk.transforms(full)
        palm_inv = np.linalg.inv(frames['palm'])
        local, _ = assets.hand.frames(q)
        for link in assets.hand.collision_links:
            worst = max(worst, float(np.max(np.abs(palm_inv@frames[link]-local[link]))))
        if any(abs(full[n]-full[driver]*mult-offset)>1e-12
               for n,(driver,mult,offset) in model.mimics.items()):
            raise ValueError('Complete-arm mimic differs from the hand calibration')
    if worst > 1e-10:
        raise ValueError('Palm-relative hand geometry differs in the combined robot')
    paths = {str(Path(config[k]).resolve()):sha(config[k]) for k in (
        'robot_urdf', 'physics_config', 'diagnostic_config', 'arm_collision_refinement', 'reference_plan')}
    return dict(independent_joint_count=len(model.active_names),
        arm_joint_count=6, wrist_joint_count=2, finger_joint_count=16,
        derived_mimic_count=len(model.mimics),
        states_checked=len(calibration['states']),
        worst_complete_robot_vs_hand_FK_entry_error=worst,
        existing_backend_configuration=config, input_sha256=paths,
        old_grasp_reexecuted=False, hardware_connections=0, physics_steps=0)


def map_frame(directory, assets, calibration, max_candidates=32, tolerance_m=.002):
    receipt_path = directory/'RESULTS.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('full_representation_forward') is not True:
        raise ValueError('Expected a real saved representation-network forward')
    source = directory/'raw_camera_candidates.npz'
    with np.load(source, allow_pickle=False) as z:
        candidates = z['candidates'].copy()
        features = z['candidate_features'].copy()
    if candidates.ndim != 2 or candidates.shape[1] != 15 or len(features) != len(candidates):
        raise ValueError('CGR candidates and features lost row alignment')
    camera = rigid(receipt['scene_binding']['T_WC'])
    # Choose representatives only to check the API, without a trained Boya
    # decision head. This order is not claimed to predict Boya success.
    indices = np.argsort(-candidates[:,0], kind='stable')[:max_candidates]
    slots = [s for s in calibration['slots'] if s['self_collision_clear']]
    rows = []
    worst = 0.
    from asset_adapter import cgr_reference_pose
    for index in indices:
        raw = candidates[index]
        for gesture_id in assets.families:
            matches = [s for s in slots if s['gesture_id']==gesture_id
                and abs(s['reference_width_m']-float(raw[1])) <= tolerance_m]
            if not matches:
                continue
            slot = min(matches, key=lambda s:(abs(s['reference_width_m']-float(raw[1])), s['slot_id']))
            palm = map_slot(raw, camera, slot)
            error = float(np.max(np.abs(palm@np.asarray(slot['palm_T_reference'])-
                cgr_reference_pose(raw,camera))))
            worst = max(worst,error)
            rows.append(dict(source_candidate_index=int(index),slot_id=slot['slot_id'],
                gesture_id=gesture_id,opposing_group=slot['opposing_group'],
                requested_width_m=float(raw[1]),reference_width_m=slot['reference_width_m'],
                width_error_m=abs(slot['reference_width_m']-float(raw[1])),
                raw_depth_m=float(raw[2]),hand_additional_depth_m=0.,thumb_side=1,
                raw_cgr_score=float(raw[0]),score_source='published CGR antipodal score; NOT a Boya hand score',
                world_T_palm=palm.tolist(),motor_names=list(assets.hand.names),
                open_motors=slot['independent_q'],closed_motors=slot['close_q'],wrist=slot['wrist'],
                FK_frame_roundtrip_error=error,
                checks=dict(limits=True,self_collision=True,scene_collision=None,table=None,arm_path=None),
                geometry_admitted=False,execution_authorized=False,label=None))
    if worst > 1e-10:
        raise ValueError('CGR-to-hand frame round trip failed')
    return dict(source_frame=str(directory),source_frame_sha256=sha(receipt_path),
        source_candidates_sha256=sha(source),source_candidate_count=len(candidates),
        source_feature_shape=list(features.shape),source_feature_row_order_preserved=True,
        source_parser_boundary=receipt.get('raw_candidate_boundary'),
        retained_original_DH3_demo_gates=True,new_network_forwards=0,
        hand_head_loaded=False,matched_requests=len(rows),
        worst_frame_roundtrip_error=worst,requests=rows,
        physics_steps=0,training_samples_generated=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canondex-root',type=Path,default=DEFAULT_CANONDEX)
    p.add_argument('--hand-urdf',type=Path,default=DEFAULT_URDF)
    p.add_argument('--hand-scene',type=Path,default=DEFAULT_SCENE)
    p.add_argument('--backend-config',type=Path,required=True)
    p.add_argument('--frames',type=Path,nargs='*',default=[])
    p.add_argument('--samples',type=int,default=33)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report=dict(schema='anydex_boya_adapter_preparation_v1',status='started',
        physics_steps=0,new_network_forwards=0,training_samples_generated=0,training_steps=0,
        Boya_pickup_success_claim=False,full_arm_execution_connected=False)
    save(args.output_dir/'RESULTS.json',report)
    assets=BoyaAssets(args.canondex_root,args.hand_urdf,args.hand_scene)
    calibration=assets.calibrate(args.samples)
    save(args.output_dir/'calibration.json',calibration)
    config=json.loads(args.backend_config.read_text())
    arm=check_full_arm(assets,calibration,config)
    save(args.output_dir/'complete_arm_interface.json',arm)
    frames=[]
    for frame in args.frames:
        result=map_frame(frame.resolve(),assets,calibration)
        save(args.output_dir/(frame.parent.name+'_requests.json'),result)
        frames.append({k:v for k,v in result.items() if k!='requests'})
    lock=assets.source_lock()
    if lock!=calibration['source_lock']:
        # Imports introduced by full-arm verification extend the code lock only;
        # the asset bytes must still be exactly the same.
        if lock['source_files']!=calibration['source_lock']['source_files']:
            raise ValueError('An external source asset changed while preparing the adapter')
    save(args.output_dir/'source_lock.json',lock)
    report.update(status='geometric_calibration_and_complete_arm_state_interface_verified',
        eligible_gestures=list(assets.families),excluded_gestures=calibration['excluded_templates'],
        calibration_states=len(calibration['states']),reference_slots=len(calibration['slots']),
        self_collision_clear_states=sum(s['self_collision_clear'] for s in calibration['states']),
        full_arm_independent_joints=arm['independent_joint_count'],
        worst_complete_robot_vs_hand_FK_entry_error=arm['worst_complete_robot_vs_hand_FK_entry_error'],
        frames=frames,elapsed_wall_s=time.monotonic()-start,
        source_asset_files=len(lock['source_files']),
        next_boundary='Fresh RGB-D -> Boya-specific candidates -> full scene/path admission -> existing native execution -> aligned labels. No trained Boya decision model yet.')
    save(args.output_dir/'RESULTS.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
