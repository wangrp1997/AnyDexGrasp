"""Read saved resident-control files only. No simulator/model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def need(value, message):
    if not value: raise ValueError(message)


def expand(q, columns, names, urdf):
    nodes={n.get('name'):n for n in ET.parse(urdf).getroot().findall('joint')}
    states={name:q[:,i] for i,name in enumerate(columns)}
    def one(name, seen):
        if name in states:return states[name]
        need(name not in seen,'Mimic cycle')
        mimic=nodes[name].find('mimic');need(mimic is not None,'Unknown target column')
        states[name]=one(mimic.get('joint'),seen|{name})*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
        return states[name]
    return np.asarray([one(name,set()) for name in names]).T.astype(np.float32)


def job(path, index, scene, pid, baseline=None):
    ready=read(path/'READY.json');plan=read(ready['plan']['path'])
    entry=read(path/'RESIDENT_EXECUTION_ENTRY.json');execution=path/'execution'
    result=read(path/'LIFT_PIPELINE_EXECUTION.json');record=read(path/'COLLECTION_RECORD.json')
    native=read(execution/'evaluation.json');lift=read(execution/'LIFT_EVALUATION.json')
    setup=read(execution/'MULTI_OBJECT_SETUP.json');initial=read(execution/'initial_state_audit.json')
    audit=read(path/'RUNTIME_AUDIT.json')
    need(read(path/'EXECUTION_STARTED.json')['pid']==pid,'Job moved to a different process')
    need(entry['status']=='complete' and result['complete'] and audit['status']=='pass','Incomplete or rejected original job')
    need(result['label']==record['label']==audit['label'],'Existing label mismatch')
    need(native['resident_scene']['reused']==(index>0),'Native asset-reuse sequence differs')
    need(native['resident_scene']['asset_setup_count']==1 and native['resident_scene']['state_reset_count']==index,'Native asset/reset count differs')
    need(native['resident_scene']['native_audit_count']==index+1,'Every job native gate not repeated')
    need(native['native_within_20um'] and setup['native']['within_20um'],'Native original 20um gate missing')
    need(all(lift['gates'].values()),'Original lift gates failed')
    need(native['state_writes_during_execution']==setup['physical_state_writes_during_execution']==lift['state_writes_during_execution']==0,'State writes inside control')
    need(setup['every_dynamic_body_weight_n']==[0.,0.,-.98] and native['object_weight_world_n']==[0.,0.,-.98],'Original permanent force differs')
    need(float(native['object_mass_kg'])==float(np.float32(.1)) and native['object_friction']==.6,'Primary native material differs')
    need(len(setup['entry'])==3 and len(setup['additional_materials'])==2,'All three body handles missing')
    positions=[];rotations=[];speeds=[]
    for obj in scene['objects']:
        key=obj['object_id'];actual=setup['entry'][key];T=np.asarray(obj['world_T_object'])
        pose=np.asarray(actual['pose_xyzw']);positions.append(float(abs(pose[:3]-T[:3,3]).max()))
        rotations.append(float(abs(Rotation.from_quat(pose[3:]).as_matrix()-T[:3,:3]).max()))
        speeds.append(float(abs(np.asarray(actual['velocity'])).max()))
        if key!=scene['target_object_id']:
            mat=setup['additional_materials'][key]
            need(mat==dict(mass_kg=.1,friction=.6,native_readback_ok=True,world_weight_n=[0.,0.,-.98]),'Nonprimary material/force readback differs')
            restore=setup['additional_initial_cold_restores'][key]
            need(restore['physics_steps']==0 and restore['after_max_error']<=2e-5,'Additional body restore did not meet original gate')
    need(max(positions)<=2e-5 and max(rotations)<=1e-4 and max(speeds)<=2e-5,'Three-body actual entry differs from bound observation')
    need(initial['errors']['max_joint_speed_rad_s']==0 and initial['errors']['max_joint_position_error_rad']<=2e-5,'Robot home reset differs')
    with np.load(ready['commands']['path'],allow_pickle=False) as z:q={k:z[k].copy() for k in z.files}
    with np.load(execution/'trace.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    expected=expand(q['joint_target'],list(q['joint_names']),list(trace['joint_names']),plan['arguments']['robot_urdf'])
    need(np.array_equal(expected,trace['joint_target']),'Actual 28 targets differ from original 24 independent commands')
    velocity=np.zeros_like(expected);arm_columns=[list(trace['joint_names']).index(name) for name in
        ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint')]
    velocity[:,arm_columns]=np.gradient(expected[:,arm_columns].astype(float),.004,axis=0,edge_order=2)
    need(np.array_equal(velocity,trace['joint_velocity_target']),'Original six-arm velocity feedforward differs')
    for name in ('phase','time_s','desired_palm'):need(np.array_equal(q[name],trace[name]),'Actual '+name+' differs')
    need(len(q['phase'])==result['physics_steps']==lift['total_physics_steps'],'Full command step count differs')
    pointer=read(execution/'MULTI_OBJECT_CONTACTS.json')
    need(pointer['lossless'] and pointer['contact_records_omitted']==0,'Lossy contacts')
    need(Path(pointer['data']['path']).stat().st_size==pointer['data']['bytes'],'Journal byte count changed')
    need(audit['audit_inputs_sha256'][pointer['data']['path']]==pointer['data']['sha256'],'First full decode audit does not bind current journal pointer')
    for source in entry['source_bindings'].values():
        need(sha(source['path'])==source['sha256'],'Resident source changed: '+source['path'])
        need(audit['audit_inputs_sha256'].get(source['path'])==source['sha256'],
             'First numeric audit omitted a resident source dependency: '+source['path'])
    bindings={name:bind(path/name) for name in ('READY.json','RESIDENT_EXECUTION_ENTRY.json','LIFT_PIPELINE_EXECUTION.json','COLLECTION_RECORD.json','RUNTIME_AUDIT.json')}
    comparison=None
    if baseline is not None:
        old_ready=read(baseline/'READY.json');old_result=read(baseline/'LIFT_PIPELINE_EXECUTION.json')
        with np.load(old_ready['commands']['path'],allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
        comparison=dict(reference_ready=bind(baseline/'READY.json'),
            reference_commands=bind(old_ready['commands']['path']),
            reference_execution=bind(baseline/'LIFT_PIPELINE_EXECUTION.json'),
            physical_arrays_equal={k:np.array_equal(q[k],old[k]) for k in
                ('initial_position','joint_names','joint_target','phase','desired_palm','time_s')},
            original_label=old_result['label'],label_equal=old_result['label']==result['label'],
            original_steps=old_result['physics_steps'],steps_equal=old_result['physics_steps']==result['physics_steps'])
    return dict(gesture_id=plan['gesture_id'],label=result['label'],physics_steps=result['physics_steps'],
        physics_clock_before_first_command=entry['physics_clock_before_first_command'],
        actual_28_targets_exact=True,original_velocity_targets_exact=True,
        phase_time_palm_exact=True,independent_runtime_audit='pass',
        reused=native['resident_scene']['reused'],asset_setups=native['resident_scene']['asset_setup_count'],
        resets=native['resident_scene']['state_reset_count'],native_audits=native['resident_scene']['native_audit_count'],
        max_entry_position_error_m=max(positions),max_entry_rotation_matrix_error=max(rotations),max_entry_speed=max(speeds),
        state_writes_during_control=0,extra_restore_integration_steps=0,
        native_gate_20um=True,all_original_lift_gates_pass=True,
        commands=bind(ready['commands']['path']),trace=bind(execution/'trace.npz'),
        source_sha_verified_by_first_audit=True,contact_journal=pointer['data'],
        first_audit_costs=audit['costs'],artifacts=bindings,reference_comparison=comparison,
        outcomes={k:{name:v[name] for name in ('task_success','final_reference_rise_m','final_source_bottom_m','trailing_observed_pickup_run_s')} for k,v in lift['objects'].items()})


def audit(root, reference=None):
    root=Path(root).resolve();protocol=read(root/'RESIDENT_COLLECTION_PROTOCOL.json')
    for source in protocol['sources'].values(): need(sha(source['path'])==source['sha256'],'Frozen source changed: '+source['path'])
    output=[]
    for directory in sorted((root/'scenes').iterdir()):
        session_dir=directory/'resident_session'
        if not session_dir.exists():continue
        receipt=read(session_dir/'SESSION_RECEIPT.json');group=read(session_dir/'EXECUTE_MANY_RESULTS.json')
        need(group['status']=='complete' and receipt['status']=='closed','Resident group not completed')
        n=len(group['jobs']);need(receipt['Kit_starts']==1 and receipt['asset_setup_count']==1,'One-App/one-asset claim differs')
        need(receipt['state_reset_count']==n-1 and receipt['scene_rebuild_count']==0,'Unexpected reset/rebuild count')
        need(receipt['execution_starts_with_unknown_steps']==receipt['added_reset_physics_steps']==receipt['retry_count']==0,'Unknown cost, integration or retry present')
        scene=read(directory/'capture/observed_scene.json')
        pid=read(session_dir/'SESSION_STARTED.json')['pid']
        jobs=[job(Path(row['prepared_dir']),i,scene,pid,
            Path(reference)/'scenes'/directory.name/Path(row['prepared_dir']).name if reference else None)
            for i,row in enumerate(group['jobs'])]
        clock_gaps=[b['physics_clock_before_first_command']-(a['physics_clock_before_first_command']+a['physics_steps'])
                    for a,b in zip(jobs,jobs[1:])]
        need(clock_gaps==[0]*(n-1),'Unaccounted integration between resident jobs')
        need(sum(j['physics_steps'] for j in jobs)==receipt['known_execution_physics_steps'],'Session step sum differs')
        output.append(dict(scene_id=scene['scene_id'],session=bind(session_dir/'SESSION_RECEIPT.json'),
            group=bind(session_dir/'EXECUTE_MANY_RESULTS.json'),Kit_starts=1,asset_setups=1,
            resets=n-1,rebuilds=0,pid=pid,all_jobs_same_process=True,executed_job_count=n,
            inter_job_physics_clock_gaps=clock_gaps,
            physics_steps=sum(j['physics_steps'] for j in jobs),jobs=jobs))
    return dict(schema='anydex_boya_resident_readonly_control_audit_v1',status='pass',
        run_dir=str(root),protocol=bind(root/'RESIDENT_COLLECTION_PROTOCOL.json'),
        sources_unchanged=True,groups=output,completed_group_count=len(output),
        Kit_starts=sum(s['Kit_starts'] for s in output),executions=sum(len(s['jobs']) for s in output),
        physics_steps=sum(s['physics_steps'] for s in output),label_count=sum(len(s['jobs']) for s in output),
        positive=sum(j['label'] for s in output for j in s['jobs']),
        negative=sum(1-j['label'] for s in output for j in s['jobs']),
        original_runtime_journals_redecoded=False,read_only=True,new_physics_steps=0,
        audit_scope='Independent saved-command expansion, native/three-body initial/reset/material/gates; first original full-contact numeric audits are read and bound, not repeated',
        exclude_training=protocol['exclude_training'],generated_unix_s=time.time())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dir',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--reference-batch',type=Path);a=p.parse_args()
    value=audit(a.run_dir,a.reference_batch);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='groups'},indent=2))
