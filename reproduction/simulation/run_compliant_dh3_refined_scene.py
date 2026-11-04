"""Saved original top1 executed with a declared compliant common-shaft proxy.

The old accepted source is inherited only for loading geometry, wrist control,
and observation. Elastic-branch topology and parameters are new engineering
assumptions. Every selected pose, scene, old/new source and physical step is
retained; this is not a reproduction of the hardware's internal dynamics.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'outputs/native_deps/python_overlay'),str(ROOT/'reproduction/runtime'),str(ROOT/'reproduction/simulation')]
import dh3_time_refinement_smoke as accepted
from dh3_time_refinement_smoke import (PhysicsAssumptions,p,sha,smooth,
                                      sampled_transmission_metrics,write_controller_receipt as rigid_unused_write_controller_receipt)
from dh3_compliant_proxy import CompliantDH3 as CoupledDH3, write_controller_receipt as accepted_write_controller_receipt

# Recorded before the generic interface was implemented; this module is reused,
# not copied or rewritten. A mismatch requires explicit review of that change.
ACCEPTED_CONTROLLER_SHA256='8d33ba1a4ca4d33df31da9717b818505bed73554df3c171696a7a5f0140df239'


def load_scene_module(observation):
    name=observation.get('scene_module','synthetic_scene')
    if name not in ('synthetic_scene','primitive_scene','mesh_scene'):
        raise ValueError('Unsupported frozen scene module: '+str(name))
    return importlib.import_module(name)


def freeze_run_sources(output_dir,receipt,scene_module):
    """Bind all executable port/scene sources before the first physical step."""
    if sha(accepted.__file__)!=ACCEPTED_CONTROLLER_SHA256:
        raise ValueError('The previously accepted controller source changed')
    source=Path(__file__).resolve();scene_source=Path(scene_module.__file__).resolve()
    copies=[]
    for path,target_name in [(source,'runner_source_frozen.py'),
                             (scene_source,'scene_source_frozen.py')]:
        target=output_dir/target_name;shutil.copyfile(path,target)
        assert sha(path)==sha(target)
        copies.append({'source':str(path),'sha256':sha(path),'frozen_copy':str(target.resolve())})
    receipt.update(runner_script=str(source),runner_script_sha256=sha(source),
        scene_module=scene_module.__name__,scene_module_script=str(scene_source),scene_module_sha256=sha(scene_source),
        source_dependencies=copies,
        execution_policy={'original_top1_only':True,'physics_duration_s':10.25,'integration_hz':3840,
                          'num_sub_steps':1,'new_network_inference':False,'new_pose_selection':False,
                          'dynamic_reset_or_added_constraint_guard':True,'explicit_compliant_proxy':True},
        accepted_controller_reference={'path':str(Path(accepted.__file__).resolve()),
                                       'sha256':ACCEPTED_CONTROLLER_SHA256,
                                       'previous_passed_result':str(ROOT/'outputs/dh3_dt1920_cube_physics_20261103_v1/RESULTS.json')})
    (output_dir/'RUN_SOURCES.json').write_text(json.dumps({
        'acquisition':'All copies and hashes recorded before first physical integration',
        'runner_and_scene':copies,'imported_controller':receipt['accepted_controller_reference'],
        'assumptions':asdict(PhysicsAssumptions(dt=1/3840))},indent=2)+'\n')


def write_controller_receipt(output_dir,receipt):
    accepted_write_controller_receipt(output_dir,receipt)
    snapshot_path=output_dir/'SOURCE_SNAPSHOT.json'
    snapshot=json.loads(snapshot_path.read_text())
    snapshot['additional_dependencies']=[
        {'path':row['frozen_copy'],'sha256':row['sha256'],'original_source':row['source']}
        for row in receipt['source_dependencies']]
    snapshot['additional_dependencies_acquisition']='Recorded and copied before first physical integration'
    snapshot_path.write_text(json.dumps(snapshot,indent=2)+'\n')


def run_scene(output_dir, manifest_path, selection_results, scene_manifest):
    """One frozen original top1, original depth/backoff, bounded scripted wrist."""
    from scipy.spatial.transform import Rotation
    from PIL import Image
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('Retain every trial; use a new directory')
    output_dir.mkdir(parents=True);(output_dir/'frames').mkdir()
    selection_results=Path(selection_results);scene_manifest=Path(scene_manifest)
    selection=json.loads(selection_results.read_text());observation=json.loads(scene_manifest.read_text())
    scene_module=load_scene_module(observation)
    non_contact_inner_iterations=20;time_step_s=1/3840
    if not selection.get('selected'):raise ValueError('Original pipeline did not select a hand pose')
    candidate_path=Path(selection['output']['path']);assert sha(candidate_path)==selection['output']['sha256']
    selected=np.load(candidate_path,allow_pickle=False)
    hand_candidate=selected['top1_DH3_grasp_array'];two_candidate=selected['top1_two_finger_grasp_array']
    assert hand_candidate.shape==(19,) and two_candidate.shape==(17,)
    T_WC=np.array(observation['camera']['T_WC']);H=np.eye(4);H[:3,:3]=hand_candidate[3:12].reshape(3,3);H[:3,3]=hand_candidate[12:15]
    T_WH=T_WC@H;direction=T_WC[:3,:3]@two_candidate[4:13].reshape(3,3)[:,0];direction/=np.linalg.norm(direction)
    registers=hand_candidate[15:17];depth=float(hand_candidate[1]);backoff=float(registers[0]/100*.005)
    target=T_WH[:3,3]+(depth-backoff)*direction;pre=target-.07*direction
    orientation=Rotation.from_matrix(T_WH[:3,:3]).as_quat().tolist()
    a=PhysicsAssumptions(non_contact_inner_iterations=non_contact_inner_iterations,dt=time_step_s);client=p.connect(p.DIRECT);start=time.monotonic();rows=[];allcontacts=[];frames=[]
    try:
        scene=(scene_module.create_scene(p,client) if scene_module.__name__=='synthetic_scene'
               else scene_module.create_scene(p,client,spec=observation['spec']))
        scene_module.assert_scene_matches_manifest(p,client,scene,observation)
        assert abs(scene['spec']['physics']['time_step_s']-1/240)<1e-15
        # Observation/geometry signature remains immutable; only integration
        # resolution is explicitly overridden after the zero-step binding.
        p.setTimeStep(a.dt,physicsClientId=client)
        p.setPhysicsEngineParameter(numNonContactInnerIterations=a.non_contact_inner_iterations,physicsClientId=client)
        hand=CoupledDH3(client,manifest_path,pre.tolist(),orientation,*registers,assumptions=a)
        receipt=hand.receipt();receipt.update(scope='One original selected-candidate scripted-wrist simulation port',
            selection_results=str(selection_results),selection_sha256=sha(selection_results),candidate_npz=str(candidate_path),candidate_sha256=sha(candidate_path),
            scene_manifest=str(scene_manifest),scene_manifest_sha256=sha(scene_manifest),scene_signature_sha256=scene['scene_signature_sha256'],
            scene_observation_bound_before_steps=True,integration_parameter_override={'observation_engine_time_step_s':scene['spec']['physics']['time_step_s'],'actual_integration_time_step_s':a.dt,'num_sub_steps':1,'scene_signature_unchanged':True,'override_before_first_integration':True},body_ids={'hand':hand.body,'object':scene['object_id'],'table':scene['table_id']},T_WC=T_WC.tolist(),original_hand_datum_T_WH=T_WH.tolist(),
            original_depth_m=depth,original_position_backoff_m=backoff,net_insertion_m=depth-backoff,
            world_approach_axis=direction.tolist(),world_pre_baseLink_m=pre.tolist(),world_target_baseLink_m=target.tolist(),
            baseLink_orientation_xyzw=orientation,rotation_matrix_orthogonality_error=float(abs(T_WH[:3,:3].T@T_WH[:3,:3]-np.eye(3)).max()),
            wrist_script={'approach_m':.07,'approach_s':3.,'close_s':2.,'close_register':0.,'hold_closed_s':.75,'lift_world_z_m':.1,'lift_s':3.,'hold_lift_s':1.,'initial_hold_s':.5},
            port_boundary='Saved original top1/depth/approach retained; single common virtual shaft with elastic branches, declared approximate dynamics. No paper hardware success-rate claim.')
        freeze_run_sources(output_dir,receipt,scene_module)
        write_controller_receipt(output_dir,receipt)
        view=p.computeViewMatrix([.35,-.4,.35],[0,.08,.12],[0,0,1]);projection=p.computeProjectionMatrixFOV(50,640/480,.01,2.)
        frame_stride=int(round(1/(20*a.dt)))
        assert abs(frame_stride*a.dt-.05)<1e-12
        initial_position=np.array(p.getBasePositionAndOrientation(scene['object_id'],physicsClientId=client)[0])
        guarded_names=['resetJointState','resetBasePositionAndOrientation','resetBaseVelocity','resetSimulation','createConstraint']
        guarded_originals={name:getattr(p,name) for name in guarded_names}
        def forbid_reset(*args,**kwargs):raise RuntimeError('No dynamic state reset or added constraint')
        for name in guarded_names:setattr(p,name,forbid_reset)
        try:
            total=10.25
            for step in range(round(total/a.dt)):
                t=step*a.dt;pos=pre.copy();position=float(registers[0]);phase='hold_open'
                if .5<=t<3.5:pos=pre+(target-pre)*smooth((t-.5)/3);phase='approach'
                elif t>=3.5:pos=target.copy();phase='close'
                if 3.5<=t<5.5:position=float(registers[0])*(1-smooth((t-3.5)/2))
                elif t>=5.5:position=0.;phase='hold_closed'
                if t>=6.25:pos=target+np.array([0,0,.1])*smooth((t-6.25)/3);phase='lift' if t<9.25 else 'hold_lift'
                targets=hand.command(position,float(registers[1]),pos.tolist(),orientation)
                p.stepSimulation(physicsClientId=client)
                row=hand.observe();object_pose=p.getBasePositionAndOrientation(scene['object_id'],physicsClientId=client)
                object_velocity=p.getBaseVelocity(scene['object_id'],physicsClientId=client)
                row.update(step=step+1,time=(step+1)*a.dt,phase=phase,register_target=np.array([position,registers[1]]),
                           joint_target=np.array([targets[n] for n in hand.movable_names]),
                           object_pose=np.r_[object_pose[0],object_pose[1]],object_velocity=np.r_[object_velocity[0],object_velocity[1]],
                           object_aabb=np.array(p.getAABB(scene['object_id'],-1,physicsClientId=client)))
                rows.append(row)
                # Contact records retain all robot/object/table/self contact costs.
                for c in p.getContactPoints(physicsClientId=client):
                    if hand.body in (c[1],c[2]):allcontacts.append([step+1,c[1],c[2],c[3],c[4],c[8],c[9],c[10],c[12]])
                if step%frame_stride==0:
                    rendered=p.getCameraImage(640,480,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,
                                               shadow=0,lightDirection=[1,1,3],physicsClientId=client)
                    filename='frame_%05d.png'%(step//frame_stride)
                    Image.fromarray(np.asarray(rendered[2],dtype=np.uint8).reshape(480,640,4)[...,:3]).save(output_dir/'frames'/filename)
                    frames.append({'path':'frames/'+filename,'physics_step':step+1,'physical_time_s':(step+1)*a.dt})
        finally:
            for name,value in guarded_originals.items():setattr(p,name,value)
        arrays={key:np.array([row[key] for row in rows]) for key in rows[0]}
        contact_array=np.asarray(allcontacts,dtype=float).reshape(-1,9)
        np.savez_compressed(output_dir/'trace.npz',**arrays);np.save(output_dir/'contacts.npy',contact_array)
        (output_dir/'frames.json').write_text(json.dumps({'fps':20,'every_physics_steps':frame_stride,'actual_simultaneous_render':True,'zero_additional_physics_steps_for_media':True,'frames':frames},indent=2)+'\n')
        finite=all(np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in 'fiu')
        lift=arrays['object_pose'][-1,2]-initial_position[2];final_low=arrays['object_aabb'][-1,0,2]
        is_object=(contact_array[:,1]==scene['object_id'])|(contact_array[:,2]==scene['object_id']) if len(contact_array) else np.array([],bool)
        is_table=(contact_array[:,1]==scene['table_id'])|(contact_array[:,2]==scene['table_id']) if len(contact_array) else np.array([],bool)
        object_contacts=contact_array[is_object];table_contacts=contact_array[is_table]
        last_contacts=object_contacts[(object_contacts[:,0]>len(rows)-round(.5/a.dt))&(object_contacts[:,6]>0)] if len(object_contacts) else object_contacts
        gearmax=float(abs(arrays['gear_residual']).max());maxspeed=float(abs(arrays['v']).max())
        torque_bounds=np.zeros(len(hand.ids))
        torque_bounds[hand.movable_names.index(hand.position_master)]=a.position_motor_max_torque_Nm
        torque_bounds[hand.movable_names.index(hand.rotation_master)]=a.rotation_motor_max_torque_Nm
        motor_bound=bool((abs(arrays['motor_torque'])<=torque_bounds+1e-8).all())
        gear_bound=bool((abs(arrays['gear_reaction'])<=a.gear_max_torque_Nm+1e-7).all())
        wrist_bound=bool((arrays['wrist_position_error']<.002).all())
        stable=bool(finite and gearmax<.01 and maxspeed<1.5 and motor_bound and gear_bound and wrist_bound)
        held=bool(lift>.05 and final_low>.04 and len(last_contacts)>0)
        result={'scope':'Saved original top1 with declared common-shaft elastic-branch proxy; not original DH3 hardware reproduction',
            'status':'object_lifted' if held and stable else 'not_lifted_or_port_invalid','lift_observed':bool(held),'port_stability_pass':stable,
            'rigid_transmission_metrics':sampled_transmission_metrics(arrays,a.gear_max_torque_Nm,a.dt),'physics_steps':len(rows),'physical_seconds':len(rows)*a.dt,'wall_seconds_including_actual_render':time.monotonic()-start,
            'object_final_lift_m':float(lift),'object_final_lowest_world_z_m':float(final_low),'object_max_lift_m':float(arrays['object_pose'][:,2].max()-initial_position[2]),
            'finite':finite,'motor_torque_bounds_pass':motor_bound,'gear_reaction_bounds_pass':gear_bound,'wrist_tracking_pass':wrist_bound,'max_gear_residual_rad':gearmax,'per_gear_max_residual_rad':dict(zip([g['dependent'] for g in hand.gears],abs(arrays['gear_residual']).max(0).tolist())),
            'peak_joint_velocity_rad_s':maxspeed,'max_wrist_COM_error_m':float(arrays['wrist_position_error'].max()),
            'peak_motor_torque_Nm':abs(arrays['motor_torque']).max(0).tolist(),'peak_gear_reaction_Nm':abs(arrays['gear_reaction']).max(0).tolist(),
            'hand_object_contact_rows':len(object_contacts),'minimum_hand_object_contact_distance_m':float(object_contacts[:,5].min()) if len(object_contacts) else None,
            'peak_hand_object_normal_force_N':float(object_contacts[:,6].max()) if len(object_contacts) else None,
            'hand_table_contact_rows':len(table_contacts),'minimum_hand_table_contact_distance_m':float(table_contacts[:,5].min()) if len(table_contacts) else None,
            'final_half_second_hand_object_contact_rows':len(last_contacts),'state_resets_after_first_step':0,'object_attachment_constraints':0,'object_gravity_retained':True,
            'continuous_same_scene':True,'actual_rendered_frames':len(frames),'normal_speed_video_fps':20,'actual_time_step_s':a.dt,'frame_stride_physics_steps':frame_stride,
            'controller_sha256':sha(output_dir/'CONTROLLER.json'),'trace_sha256':sha(output_dir/'trace.npz'),
            'runner_sha256':sha(__file__),'imported_controller_sha256':sha(accepted.__file__),
            'scene_module':scene_module.__name__,'network_rerun':False,'candidate_reselection':False,
            'controller_kind':'declared_common_shaft_elastic_branch_proxy_v1','approximation_not_paper_dynamics':True,'boundary':'Engineering elastic-branch approximation with common bounded input and logged spring/shaft states; original 13 entities retained. Not identified DH3 differential or spring mechanism.'}
        result.update(shaft_peak_motor_torque_Nm=float(abs(arrays['shaft_motor_torque_Nm']).max()),shaft_peak_velocity_rad_s=float(abs(arrays['shaft_v_rad_s']).max()),branch_peak_transmitted_torque_Nm=abs(arrays['branch_torque_Nm']).max(0).tolist(),branch_peak_elastic_energy_J=float(arrays['branch_elastic_energy_J'].max()),independent_finger_servos=False)
        (output_dir/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
    finally:p.disconnect(client)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--adapter-manifest',type=Path,default=ROOT/'outputs/dh3_geometry_adapter_20261103_v2/ADAPTER.json')
    ap.add_argument('--scene-manifest',type=Path,required=True)
    ap.add_argument('--selection-results',type=Path,required=True,
                    help='Unchanged original top1 selection RESULTS.json; binds its NPZ by SHA')
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    try:
        run_scene(args.output_dir.resolve(),args.adapter_manifest.resolve(),args.selection_results.resolve(),args.scene_manifest.resolve())
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir/'RESULTS.json').exists():
            (args.output_dir/'FAILED.json').write_text(json.dumps({
                'status':'execution_error','type':type(exc).__name__,'error':str(exc),
                'traceback':traceback.format_exc(),'results_not_completed':True},indent=2)+'\n')
        raise


if __name__=='__main__':main()
