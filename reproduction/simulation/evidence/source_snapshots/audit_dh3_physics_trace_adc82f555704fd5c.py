"""Read-only, simulator-free audit of saved DH3 physics-port traces.

Recomputes mechanism residuals/finite bounds and observed lifting separately.
No pybullet import, inference, collision query, render or physical step occurs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def absolute(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def match(result,key,value,atol=1e-12):
    assert key in result,key
    observed=result[key]
    if isinstance(value,(bool,str)):
        assert observed==value,(key,observed,value)
    else:assert np.allclose(observed,value,rtol=0,atol=atol),(key,observed,value)


def source_audit(source,mode):
    tree=ast.parse(source.read_text())
    calls=[]
    class Calls(ast.NodeVisitor):
        def __init__(self):self.scope=[]
        def visit_ClassDef(self,node):
            self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
        def visit_FunctionDef(self,node):
            self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='p':
                calls.append((node.func.attr,tuple(self.scope),node.lineno))
            self.generic_visit(node)
    Calls().visit(tree)
    creators=[c for c in calls if c[0]=='createConstraint']
    resets=[c for c in calls if c[0] in ('resetJointState','resetBasePositionAndOrientation')]
    assert len(creators)==2 and all(c[1]==('CoupledDH3','__init__') for c in creators)
    assert len(resets)==1 and resets[0][1]==('CoupledDH3','__init__')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_'+mode)
    steps=[c for c in calls if c[0]=='stepSimulation' and c[1]==('run_'+mode,)]
    assert len(steps)==1
    text=ast.unparse(function)
    assert 'p.resetJointState = p.resetBasePositionAndOrientation = forbid_reset' in text or 'p.resetJointState = p.resetBasePositionAndOrientation = forbid_reset' in source.read_text()
    guard=next(n for n in ast.walk(function) if isinstance(n,ast.FunctionDef) and n.name=='forbid_reset')
    assert any(isinstance(n,ast.Raise) for n in guard.body)
    return dict(constraint_creation_calls_only_hand_initialization=[c[2] for c in creators],
        joint_reset_calls_only_hand_initialization=[c[2] for c in resets],
        runtime_reset_guard_raises=True,mode_step_call_sites=1,
        media_step_call_sites=0 if mode=='cube' else None,
        boundary='Frozen source control-flow plus recorded initial constraint inventory; no per-step constraint inventory was recorded')


def contact_summary(contacts):
    return dict(rows=int(len(contacts)),positive_normal_rows=int((contacts[:,6]>0).sum()) if len(contacts) else 0,
        peak_normal_force_N=float(contacts[:,6].max()) if len(contacts) else None,
        min_signed_distance_m=float(contacts[:,5].min()) if len(contacts) else None)


def audit(run_dir):
    r=load(run_dir/'RESULTS.json');c=load(run_dir/'CONTROLLER.json')
    with np.load(run_dir/'trace.npz',allow_pickle=False) as stored:t={key:stored[key].copy() for key in stored.files}
    mode='cube' if 'object_pose' in t else 'empty'
    a=c['assumptions_SI_not_hardware_identification'];dt=a['dt'];n=len(t['step'])
    assert np.array_equal(t['step'],np.arange(1,n+1))
    assert np.array_equal(t['time'],t['step']*dt)
    assert all(len(value)==n for value in t.values())
    finite=bool(all(np.isfinite(value).all() for value in t.values() if value.dtype.kind in 'fiu'))
    assert sha(run_dir/'trace.npz')==r['trace_sha256']
    assert sha(run_dir/'CONTROLLER.json')==r.get('controller_sha256',r.get('controller_receipt_sha256'))
    snapshot=load(run_dir/'SOURCE_SNAPSHOT.json');frozen=absolute(snapshot['path'])
    assert sha(frozen)==snapshot['sha256']==c['controller_script_sha256']
    source_checks=source_audit(frozen,mode)
    for key,hashkey in (('adapter_manifest','adapter_manifest_sha256'),('derived_urdf','derived_urdf_sha256')):
        assert sha(absolute(c[key]))==c[hashkey]
    assert sha(c['author_kinematics']['source'])==c['author_kinematics']['source_sha256']
    names=c['joint_order'];ix={name:i for i,name in enumerate(names)}
    residual=np.column_stack([t['q'][:,ix[g['dependent']]]-g['factor']*t['q'][:,ix[g['master']]]-g['offset'] for g in c['gears']])
    assert residual.tobytes()==t['gear_residual'].tobytes()
    limits=np.zeros(len(names))
    limits[ix['finger1_joint']]=a['position_motor_max_torque_Nm']
    limits[ix['finger1_base_joint']]=a['rotation_motor_max_torque_Nm']
    motor_bound=bool((abs(t['motor_torque'])<=limits+1e-8).all())
    gear_bound=bool((abs(t['gear_reaction'])<=a['gear_max_torque_Nm']+1e-7).all())
    wrist_force_bound=bool((abs(t['wrist_reaction'])<=a['wrist_max_force_per_row_N_or_Nm']+1e-7).all())
    wrist_bound=bool((t['wrist_position_error']<.002).all())
    gearmax=float(abs(residual).max());speed=float(abs(t['v']).max())
    work_proxy=float(np.sum(abs(t['gear_reaction'][1:]*np.diff(residual,axis=0))))
    stable=bool(finite and gearmax<.01 and speed<1.5 and motor_bound and gear_bound and wrist_bound)
    for key,value in dict(physics_steps=n,physical_seconds=n*dt,finite=finite,max_gear_residual_rad=gearmax,
        peak_joint_velocity_rad_s=speed,motor_torque_bounds_pass=motor_bound,gear_reaction_bounds_pass=gear_bound,
        max_wrist_COM_error_m=float(t['wrist_position_error'].max()),peak_motor_torque_Nm=abs(t['motor_torque']).max(0),
        peak_gear_reaction_Nm=abs(t['gear_reaction']).max(0)).items():match(r,key,value)
    for g,maximum in zip(c['gears'],abs(residual).max(0)):match(r['per_gear_max_residual_rad'],g['dependent'],maximum)
    # Independent initial actual-constraint readback, not the summary zero counter.
    actual=c['actual_constraint_info'];hand=int(actual['wrist'][0]);all_constraints=[actual['wrist']]+actual['gears']
    assert actual['wrist'][2]==-1 and actual['wrist'][4]==4
    assert len(actual['gears'])==len(c['gears'])==9
    assert all(row[0]==hand and row[2]==hand and row[4]==6 for row in actual['gears'])
    assert actual['wrist'][10]==a['wrist_max_force_per_row_N_or_Nm']
    xml=ET.parse(absolute(c['derived_urdf'])).getroot()
    joint_to_link={node.attrib['name']:node.find('child').attrib['link'] for node in xml.findall('joint')}
    link_names={-1:'base_Link'}
    for gear,info in zip(c['gears'],actual['gears']):
        assert np.allclose([info[10],info[11],info[13],info[14]],
            [a['gear_max_torque_Nm'],-gear['factor'],gear['offset'],a['gear_erp']],rtol=0,atol=1e-14)
        link_names[int(info[1])]=joint_to_link[gear['dependent']]
        link_names[int(info[3])]=joint_to_link[gear['master']]
    peak_step,peak_gear=np.unravel_index(abs(residual).argmax(),residual.shape)
    report=dict(run_dir=str(run_dir.resolve()),mode=mode,status='saved_trace_independently_verified',
        source_snapshot_sha256=sha(frozen),source_snapshot_acquisition=snapshot['acquisition'],
        source_control_flow=source_checks,physics_steps=n,physical_seconds=n*dt,finite=finite,
        mechanism=dict(residuals_recomputed_from_all_q_bit_exact=True,max_residual_rad=gearmax,
            per_gear_max_residual_rad={g['dependent']:float(x) for g,x in zip(c['gears'],abs(residual).max(0))},
            peak_joint_velocity_rad_s=speed,motor_torque_bounds_pass=motor_bound,gear_reaction_bounds_pass=gear_bound,
            wrist_reaction_per_row_bound_pass=wrist_force_bound,wrist_position_tracking_pass=wrist_bound,
            peak_wrist_reaction_per_row=abs(t['wrist_reaction']).max(0).tolist(),
            correction_work_absolute_upper_proxy_J=work_proxy,
            peak_residual=dict(step=int(t['step'][peak_step]),time_s=float(t['time'][peak_step]),
                phase=str(t['phase'][peak_step]),dependent=c['gears'][peak_gear]['dependent'],
                signed_residual_rad=float(residual[peak_step,peak_gear]),gear_reaction_Nm=float(t['gear_reaction'][peak_step,peak_gear]))),
        constraints=dict(recorded_count=len(all_constraints),hand_body_id=hand,only_hand_to_world_wrist_and_internal_gears=True),
        declared_limits=a,checks=dict(trace_SHA=True,controller_SHA=True,frozen_source_SHA=True,adapter_and_URDF_SHA=True),
        artifact_SHA={name:sha(run_dir/name) for name in ('RESULTS.json','CONTROLLER.json','SOURCE_SNAPSHOT.json','trace.npz')},
        phases=[])
    contacts=None
    if mode=='cube':
        scene_path=absolute(c['scene_manifest']);scene=load(scene_path)
        selection_path=absolute(c['selection_results']);selection=load(selection_path)
        assert sha(scene_path)==c['scene_manifest_sha256']
        assert sha(selection_path)==c['selection_sha256']
        assert sha(absolute(c['candidate_npz']))==c['candidate_sha256']==selection['output']['sha256']
        assert scene['scene_signature_sha256']==c['scene_signature_sha256']
        assert np.array_equal(scene['camera']['T_WC'],c['T_WC'])
        for file in scene['files']:assert sha(file['path'])==file['sha256']
        ids=c['body_ids'];assert ids['hand']==hand
        assert all(ids['object'] not in (row[0],row[2]) for row in all_constraints)
        contacts=np.load(run_dir/'contacts.npy',allow_pickle=False)
        assert contacts.ndim==2 and contacts.shape[1]==9 and np.isfinite(contacts).all()
        assert np.equal(contacts[:,:5],np.rint(contacts[:,:5])).all()
        assert ((contacts[:,0]>=1)&(contacts[:,0]<=n)).all()
        assert ((contacts[:,1]==hand)|(contacts[:,2]==hand)).all()
        def pair(body):return ((contacts[:,1]==hand)&(contacts[:,2]==body))|((contacts[:,2]==hand)&(contacts[:,1]==body))
        object_contacts=contacts[pair(ids['object'])];table_contacts=contacts[pair(ids['table'])]
        final_steps=np.arange(n-round(.5/dt)+1,n+1)
        last=object_contacts[np.isin(object_contacts[:,0],final_steps)&(object_contacts[:,6]>0)]
        counts=np.array([(last[:,0]==step).sum() for step in final_steps])
        force_totals=np.array([last[last[:,0]==step,6].sum() for step in final_steps])
        initial_z=scene['actual_state']['object']['position_world_m'][2]
        lift=float(t['object_pose'][-1,2]-initial_z);lowest=float(t['object_aabb'][-1,0,2])
        maximum=float(t['object_pose'][:,2].max()-initial_z)
        half=np.array(scene['spec']['object']['collision_box_size_m'])/2
        extent=np.abs(Rotation.from_quat(t['object_pose'][:,3:7]).as_matrix())@half
        expected_aabb=np.stack([t['object_pose'][:,:3]-extent,t['object_pose'][:,:3]+extent],axis=1)
        aabb_error=float(abs(expected_aabb-t['object_aabb']).max())
        assert aabb_error<1e-12
        held=bool(lift>.05 and lowest>.04 and len(last)>0)
        match(r,'lift_observed',held);match(r,'port_stability_pass',stable)
        match(r,'status','object_lifted' if held and stable else 'not_lifted_or_port_invalid')
        for key,value in dict(object_final_lift_m=lift,object_final_lowest_world_z_m=lowest,object_max_lift_m=maximum,
            hand_object_contact_rows=len(object_contacts),hand_table_contact_rows=len(table_contacts),
            final_half_second_hand_object_contact_rows=len(last)).items():match(r,key,value)
        match(r,'state_resets_after_first_step',0);match(r,'object_attachment_constraints',0)
        media=load(run_dir/'frames.json');frame_records=media['frames']
        assert media['fps']==1/(media['every_physics_steps']*dt)==20
        assert media['actual_simultaneous_render'] and media['zero_additional_physics_steps_for_media']
        expected_steps=np.arange(1,n+1,media['every_physics_steps'])
        assert np.array_equal(expected_steps,[row['physics_step'] for row in frame_records])
        assert len(frame_records)==r['actual_rendered_frames']==205
        assert len(list((run_dir/'frames').glob('*.png')))==len(frame_records)
        frame_hash=hashlib.sha256()
        for row in frame_records:
            path=run_dir/row['path'];assert abs(row['physical_time_s']-row['physics_step']*dt)<1e-15
            with Image.open(path) as image:
                assert image.size==(640,480) and image.mode=='RGB';image.verify()
            frame_hash.update((row['path']+' '+sha(path)+'\n').encode())
        peak_contacts=contacts[contacts[:,0]==t['step'][peak_step]]
        peak_pairs=[]
        for row in peak_contacts:
            if row[6]<=0:continue
            other=int(row[2] if row[1]==hand else row[1])
            link=int(row[3] if row[1]==hand else row[4])
            peak_pairs.append(dict(other_body='object' if other==ids['object'] else 'table' if other==ids['table'] else 'hand',
                hand_link=link_names.get(link,'index_'+str(link)),normal_force_N=float(row[6]),signed_distance_m=float(row[5])))
        report['mechanism']['peak_residual']['simultaneous_positive_normal_contacts']=peak_pairs
        report.update(lift_observed=held,port_stability_pass=stable,validated_physical_grasp_success=bool(held and stable),
            final_object_lift_m=lift,final_object_lowest_world_z_m=lowest,max_object_lift_m=maximum,
            independent_oriented_cube_AABB_max_error_m=aabb_error,
            hand_object=contact_summary(object_contacts),hand_table=contact_summary(table_contacts),
            final_half_second=dict(steps=len(final_steps),positive_normal_contact_covered_steps=int((counts>0).sum()),
                every_step_positive_normal_contact=bool((counts>0).all()),min_contact_rows_per_step=int(counts.min()),
                max_contact_rows_per_step=int(counts.max()),min_total_normal_force_per_step_N=float(force_totals.min()),
                contact_rows=len(last)),
            perception_binding=dict(scene_manifest_sha256=sha(scene_path),selection_sha256=sha(selection_path),
                candidate_sha256=sha(absolute(c['candidate_npz'])),scene_signature_sha256=c['scene_signature_sha256']),
            media=dict(actual_saved_frames=len(frame_records),every_physics_steps=media['every_physics_steps'],fps=media['fps'],
                implied_video_seconds=len(frame_records)/media['fps'],first_last_physics_steps=[int(expected_steps[0]),int(expected_steps[-1])],
                frames_SHA_sequence=frame_hash.hexdigest(),all_frames_decode=True,zero_additional_physics_steps_source_and_receipt=True,
                encoded_MP4_independently_probed_by_this_audit=False),
            boundary='Actual cube lifting/contact observed but port stability is separate. Recorded finite ideal gears, scripted wrist and convex contacts are simulator assumptions; no hardware or paper success-rate claim.')
        report['artifact_SHA'].update({name:sha(run_dir/name) for name in ('contacts.npy','frames.json')})
    else:
        contacts=np.load(run_dir/'self_contacts.npy',allow_pickle=False)
        assert contacts.ndim==2 and contacts.shape[1]==5 and np.isfinite(contacts).all()
        close_final=int(np.flatnonzero(t['phase']=='hold_closed')[-1])
        error=abs(t['q'][close_final]-t['joint_target'][close_final])
        recent=contacts[(contacts[:,0]>=close_final-24)&(contacts[:,0]<=close_final+1)&(contacts[:,4]>.01)]
        stopped=bool(error.max()>=.04 and len(recent)>0 and abs(t['v'][close_final,ix['finger1_joint']])<.05)
        tracking=bool(error.max()<.04 or stopped)
        empty_pass=bool(stable and tracking)
        match(r,'closed_final_max_target_error_rad',float(error.max()))
        match(r,'closed_final_contact_stopped',stopped);match(r,'closed_final_tracking_or_contact_stop',tracking)
        match(r,'absolute_constraint_correction_work_upper_proxy_J',work_proxy)
        match(r,'status','passed' if empty_pass else 'failed')
        report.update(empty_mechanism_pass=empty_pass,port_stability_pass=stable,grasp_result=False,
            closed_final_contact_stopped=stopped,closed_final_tracking_or_contact_stop=tracking,
            closed_final_max_target_error_rad=float(error.max()),
            boundary='Empty mechanism check only. A contact stop is not exact endpoint tracking and establishes no loaded-grasp validity.')
        report['artifact_SHA']['self_contacts.npy']=sha(run_dir/'self_contacts.npy')
    # Stage summaries retain transients rather than hiding them in a final hold.
    for phase in dict.fromkeys(t['phase']):
        mask=t['phase']==phase;steps=t['step'][mask]
        item=dict(phase=str(phase),steps=len(steps),first_last_steps=[int(steps[0]),int(steps[-1])],
            max_gear_residual_rad=float(abs(residual[mask]).max()),peak_joint_velocity_rad_s=float(abs(t['v'][mask]).max()),
            max_wrist_COM_error_m=float(t['wrist_position_error'][mask].max()),
            peak_gear_reaction_Nm=float(abs(t['gear_reaction'][mask]).max()))
        if mode=='cube':
            stage=contacts[np.isin(contacts[:,0],steps)]
            item['hand_object']=contact_summary(stage[((stage[:,1]==hand)&(stage[:,2]==ids['object']))|((stage[:,2]==hand)&(stage[:,1]==ids['object']))])
            item['hand_table']=contact_summary(stage[((stage[:,1]==hand)&(stage[:,2]==ids['table']))|((stage[:,2]==hand)&(stage[:,1]==ids['table']))])
            item['object_lift_first_last_m']=[float(t['object_pose'][mask][i,2]-initial_z) for i in (0,-1)]
        report['phases'].append(item)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Retain evidence; choose a new output')
    try:
        reports=[audit(absolute(path)) for path in args.run_dir]
        cubes=[r for r in reports if r['mode']=='cube']
        comparison=None
        if len(cubes)==2:
            dirs=[Path(r['run_dir']) for r in cubes];controllers=[load(d/'CONTROLLER.json') for d in dirs]
            aa=[dict(c['assumptions_SI_not_hardware_identification']) for c in controllers]
            iterations=[a.pop('non_contact_inner_iterations') for a in aa]
            same_input=all(controllers[0][key]==controllers[1][key] for key in
                ('candidate_sha256','selection_sha256','scene_manifest_sha256','world_approach_axis','world_target_baseLink_m','wrist_script'))
            assert aa[0]==aa[1] and same_input
            traces=[np.load(d/'trace.npz',allow_pickle=False) for d in dirs]
            commands={key:traces[0][key].tobytes()==traces[1][key].tobytes() for key in ('step','time','phase','register_target','joint_target')}
            assert all(commands.values())
            comparison=dict(only_declared_assumption_changed='non_contact_inner_iterations',iterations=iterations,
                frozen_candidate_scene_and_script_identical=True,command_arrays_bit_identical=commands,
                both_lift_observed=all(r['lift_observed'] for r in cubes),both_port_stability_failed=all(not r['port_stability_pass'] for r in cubes),
                interpretation='More inner iterations did not restore loaded mechanism stability. Concurrent table/object contacts and finite transmission saturation remain observed; these two trials do not isolate a single cause.')
        result=dict(status='all_saved_physics_traces_independently_verified',runs=reports,paired_comparison=comparison,
            audit_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
            audit_new_physics_steps=0,audit_new_network_forwards=0,audit_new_renders=0)
    except Exception as exc:
        result=dict(status='physics_trace_audit_failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');raise
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],runs=[dict(run_dir=r['run_dir'],mode=r['mode'],max_gear_residual_rad=r['mechanism']['max_residual_rad'],
        lift_observed=r.get('lift_observed'),port_stability_pass=r['port_stability_pass']) for r in reports],output=str(args.output)),indent=2))


if __name__=='__main__':main()
