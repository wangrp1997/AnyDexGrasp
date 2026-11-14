"""Read-only all-body lift/contact/identity audit for new collection records.

No planner, model, native simulator or training module is imported. The audit
never creates a label: it certifies (or rejects) the already persisted record.
Both original visual identity and converted execution geometry are bound.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

TASK = 'declared_boya_clutter_lift_v1'
SCHEMA = 'anydex_boya_author_collection_runtime_audit_v1'
POINTER_SCHEMA = 'anydex_boya_contact_journal_pointer_v1'
PHASES = ['settle','home_to_pregrasp','approach','close','hold','lift']
GATES = ('joint_limits','mimics','penetration','palm_tracking','no_arm_object_support',
         'fixed_base','complete_execution','all_dynamic_contacts',
         'all_dynamic_objects_no_arm_support','author_phase_order','no_runtime_error','no_early_physical_stop')
DT = .004


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def need(condition,reason):
    if not condition:raise ValueError(reason)


def _array(value,shape=None):
    a=np.asarray(value,dtype=float)
    need((shape is None or a.shape==shape) and np.isfinite(a).all(),'Finite aligned numeric array required')
    return a


class Bindings:
    def __init__(self):self.files={}
    def add(self,path,expected=None):
        path=str(Path(path).resolve());digest=sha(path)
        need(expected is None or digest==expected,'Artifact SHA changed: '+path)
        need(path not in self.files or self.files[path]==digest,'Conflicting artifact binding: '+path)
        self.files[path]=digest
        return dict(path=path,sha256=digest)
    def tree(self,value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:self.add(value['path'],value['sha256'])
            for item in value.values():self.tree(item)
        elif isinstance(value,list):
            for item in value:self.tree(item)
    def mapping(self,value):
        for path,digest in value.items():self.add(path,digest)
    def verify(self):
        for path,digest in self.files.items():need(sha(path)==digest,'Audit input changed during read: '+path)


def iter_contact_rows(path,bindings=None):
    """Stream the explicit JSONL-gzip pointer or the historical rows JSON.

    All records, including zero-force points, are decoded. Memory is bounded
    by one tick (rather than by the number of ticks or contact points).
    """
    path=Path(path)
    if bindings is not None:bindings.add(path)
    with path.open('r') as stream:prefix=stream.read(4096)
    # The pointer is small and explicitly names its format, never fake rows.
    if '"schema"' in prefix and POINTER_SCHEMA in prefix:
        pointer=read(path)
        need(pointer.get('schema')==POINTER_SCHEMA and pointer.get('compression')=='gzip'
             and pointer.get('record_format')=='jsonl' and 'rows' not in pointer
             and pointer.get('lossless') is True and pointer.get('contact_records_omitted')==0,
             'Explicit complete lossless contact pointer required')
        data=pointer['data'];data_path=Path(data['path'])
        need(data_path.is_absolute() and data_path.stat().st_size==data['bytes'],'Contact data bytes/path changed')
        if bindings is not None:bindings.add(data_path,data['sha256']);bindings.mapping(pointer.get('source_bindings',{}))
        else:need(sha(data_path)==data['sha256'],'Contact data SHA changed')
        count=0
        with gzip.open(data_path,'rt') as stream:
            for line in stream:
                need(bool(line.strip()),'Empty contact JSONL record')
                row=json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
                count+=1;yield row
        need(type(pointer.get('record_count')) is int and count==pointer['record_count'],'Contact record count changed')
        return
    decoder=json.JSONDecoder(parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    with path.open('r') as stream:
        buffer='';started=False;done=False
        while not done:
            block=stream.read(1<<16)
            if block:buffer+=block
            if not started:
                index=buffer.find('[')
                need(index>=0 or bool(block),'Missing contact rows array')
                if index<0:continue
                need(buffer[:index].strip().replace(' ','').replace('\n','')=='{"rows":',
                     'Historical contact document must contain only rows')
                buffer=buffer[index+1:];started=True
            while True:
                buffer=buffer.lstrip()
                if buffer.startswith(','):buffer=buffer[1:].lstrip()
                if buffer.startswith(']'):
                    buffer=buffer[1:];done=True;break
                if not buffer:break
                try:row,end=decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    need(bool(block),'Truncated contact JSON');break
                yield row;buffer=buffer[end:]
            need(done or bool(block),'Truncated contact rows')
        need((buffer+stream.read()).strip()=='}','Trailing contact JSON content')


def _identity(scene,split_manifest,bindings):
    need(scene.get('schema')=='anydex_boya_author_lift_scene_v1' and scene.get('task')==TASK
         and scene.get('bin','missing') is None and scene.get('dt_s')==DT,'Explicit bin-free lift scene required')
    objects=scene['objects'];ids=[o['object_id'] for o in objects]
    need(len(ids)==3 and len(set(ids))==3 and scene.get('original_object_ids')==ids
         and scene.get('target_object_id') in ids,'All three original body identities required')
    need(split_manifest is not None,'Explicit original source split manifest required')
    bindings.add(split_manifest);manifest=read(split_manifest)['internal_mesh_groups']
    groups={};digests={}
    for name,split in (('fit','fit'),('internal_check','internal')):
        for row in manifest[name]:
            key,digest=row['object_id'],row['mesh_sha256']
            need(key not in groups and digest not in digests,'Duplicate/cross-split source identity in manifest')
            groups[key]=(split,digest);digests[digest]=split
    split=scene.get('dataset_split');need(split in ('fit','internal'),'Declared whole-scene split required')
    original=[];converted=[]
    for obj in objects:
        identity=obj.get('source_visual_mesh_sha256')
        need(obj['object_id'] in groups and groups[obj['object_id']]==(split,identity),
             'Every original scene body must belong to this source-identity split')
        need(obj.get('mesh_group_sha256')==identity,'Original visual/group identity mismatch')
        bindings.add(obj['source_visual_mesh'],identity)
        bindings.add(obj['source_mesh'],obj['source_mesh_sha256'])
        need(Path(obj['source_mesh']).resolve()==(Path(obj['object_root'])/'mesh/simplified.obj').resolve(),
             'Measured source mesh must be the actual execution object mesh')
        need(obj.get('mass_kg')==.1 and obj.get('friction')==.6 and np.isfinite(obj['scale']) and obj['scale']>0,
             'Original all-body mass/friction/scale required')
        if obj.get('collision_refinement'):bindings.add(obj['collision_refinement'],obj['collision_refinement_sha256'])
        original.append(identity);converted.append(obj['source_mesh_sha256'])
    need(scene.get('all_source_mesh_sha256')==original,'Whole-scene original identity list mismatch')
    return ids,original,converted,split


def _robot_xml(plan,bindings):
    args=plan['arguments'];bindings.add(args['robot_urdf']);bindings.add(args['hand_urdf'])
    root=ET.parse(args['robot_urdf']).getroot();joints={}
    for node in root.findall('joint'):
        if node.get('type')=='fixed':continue
        mimic=node.find('mimic');limit=node.find('limit')
        joints[node.get('name')]=dict(mimic=(None if mimic is None else
            (mimic.get('joint'),float(mimic.get('multiplier','1')),float(mimic.get('offset','0')))),
            limits=(-np.inf,np.inf) if limit is None else (float(limit.get('lower','-inf')),float(limit.get('upper','inf'))))
    # Source hand descendants begin at palm; wrist/forearm links are excluded.
    hand=ET.parse(args['hand_urdf']).getroot();hand_links={'palm'}
    for _ in range(len(hand.findall('joint'))+1):
        for joint in hand.findall('joint'):
            if joint.find('parent').get('link') in hand_links:hand_links.add(joint.find('child').get('link'))
    fixed={joint.find('child').get('link'):joint.find('parent').get('link')
           for joint in root.findall('joint') if joint.get('type')=='fixed'}
    def owner(link):
        seen=set()
        while link in fixed:
            need(link not in seen,'Cyclic fixed attachment');seen.add(link);link=fixed[link]
        return link
    return joints,{owner(link) for link in hand_links}


def _commands(directory,ready,plan,candidate,binding,bindings):
    need(json_sha({k:v for k,v in candidate.items() if k!='candidate_sha256'})==candidate['candidate_sha256'],
         'Candidate digest mismatch')
    need(json_sha({k:v for k,v in binding.items() if k!='command_binding_sha256'})==binding['command_binding_sha256'],
         'Command binding digest mismatch')
    need(candidate.get('task')==TASK and binding.get('execution_spec',{}).get('task')==TASK
         and binding.get('action_contract_validated') is True,'Actual validated lift action required')
    need(binding['commands_sha256']==ready['commands']['sha256']==plan['commands_sha256']
         and binding['candidate_sha256']==candidate['candidate_sha256']==ready['collection_candidate_sha256']
         and plan['author_collection_binding']['collection_candidate_sha256']==candidate['candidate_sha256'],
         'Candidate/native command echo mismatch')
    need(Path(binding['commands_path']).resolve()==Path(ready['commands']['path']).resolve(),'Bound actual command path mismatch')
    with np.load(ready['commands']['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    need(str(arrays['collection_candidate_sha256'].item())==candidate['candidate_sha256']
         and str(arrays['collection_execution_spec_sha256'].item())==json_sha(binding['execution_spec']),
         'Native command metadata mismatch')
    names=list(map(str,arrays['joint_names']));phases=list(map(str,arrays['phase']));n=len(phases)
    need(len(names)==24 and len(set(names))==24 and arrays['joint_target'].shape==(n,24)
         and arrays['desired_palm'].shape==(n,4,4),'Original 24 independent command columns required')
    for key in ('joint_target','initial_position','time_s','desired_palm'):_array(arrays[key])
    need([p for i,p in enumerate(phases) if i==0 or p!=phases[i-1]]==PHASES,'Ordered complete original lift phases required')
    counts={phase:phases.count(phase) for phase in PHASES}
    need(counts['settle']==50 and counts['home_to_pregrasp']>0 and counts['approach']==500
         and counts['close']==250 and counts['hold']==50 and counts['lift']>=500,
         'Original phase durations changed or lift truncated')
    need(np.array_equal(arrays['time_s'],np.arange(1,n+1)*DT),'Native command clock changed')
    need(plan.get('task')==TASK and plan.get('complete') is True and plan.get('admitted') is True
         and plan.get('approach_only') is False and not plan.get('author_transport')
         and plan['arguments'].get('pregrasp_distance_m')==.04,'Explicit source4cm full lift plan required')
    for key,name in (('path_geometry_sha256','path_geometry.json'),('lift_arm_geometry_sha256','lift_arm_geometry.json')):
        path=Path(ready['plan']['path']).parent/name;bindings.add(path,plan[key]);need(read(path).get('ok') is True,'All-body geometry not admitted')
    features=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
    need(features.shape==(480,) and np.isfinite(features).all()
         and hashlib.sha256(features.tobytes()).hexdigest()==candidate['canonical_cgr480_identity']['c_order_bytes_sha256']
         and candidate['canonicalization'].get('applied_once') is True,'Once-canonicalized float32 CGR480 mismatch')
    need(type(candidate['raw_depth_index']) is int and type(candidate['extra_depth_index']) is int
         and 0<=candidate['raw_depth_index']<5 and 0<=candidate['extra_depth_index']<4,'Original raw-depth/extra output index required')
    return arrays,names,phases,counts


def _full_target(columns,values,joints,names):
    positions={name:values[:,i] for i,name in enumerate(columns)}
    def expand(name,seen):
        if name in positions:return positions[name]
        need(name in joints and name not in seen and joints[name]['mimic'] is not None,'Missing/cyclic actual joint coordinate')
        driver,mult,offset=joints[name]['mimic'];positions[name]=expand(driver,seen|{name})*mult+offset
        return positions[name]
    return np.asarray([expand(name,set()) for name in names]).T.astype(np.float32)


def audit_runtime(scene,plan,arrays,execution,evaluation,bindings):
    """Independent numeric audit; all meshes and contact points remain read-only."""
    from scipy.spatial.transform import Rotation
    import trimesh
    ids=[o['object_id'] for o in scene['objects']]
    setup_path=execution/'MULTI_OBJECT_SETUP.json';bindings.add(setup_path);setup=read(setup_path)
    need(set(setup['body_paths'])==set(ids) and len(set(setup['body_paths'].values()))==len(ids),'Distinct native paths for all source bodies required')
    need(setup.get('physical_state_writes_during_execution')==0 and setup.get('global_gravity_m_s2')==[0.,0.,0.]
         and setup.get('every_dynamic_body_weight_n')==[0.,0.,-.98],'Original no-reset/all-body downward weight required')
    need(setup.get('native',{}).get('within_20um') is True,'All-body native import audit missing')
    need(not any('Bin' in p for p in setup['contact_targets']),'Lift runtime contains bin collision geometry')
    for key in ids:
        entry=setup['entry'][key];pose=_array(entry['pose_xyzw'],(7,));velocity=_array(entry['velocity'],(6,))
        wanted=_array(next(o for o in scene['objects'] if o['object_id']==key)['world_T_object'],(4,4))
        need(np.max(np.abs(pose[:3]-wanted[:3,3]))<=2e-5 and np.max(np.abs(Rotation.from_quat(pose[3:]).as_matrix()-wanted[:3,:3]))<=1e-4
             and np.max(np.abs(velocity))<=2e-5,'Native cold entry differs from bound observation')
    trace_path=execution/'MULTI_OBJECT_TRACE.json';bindings.add(trace_path,evaluation['trace_sha256']);rows=read(trace_path)['rows']
    n=len(arrays['phase']);need(len(rows)==n and evaluation.get('total_physics_steps')==n
         and evaluation.get('dynamic_object_steps')==n*len(ids),'Full one-tick/all-body execution counts required')
    phases=list(map(str,arrays['phase']));vertices={}
    for obj in scene['objects']:
        mesh=trimesh.load_mesh(obj['source_mesh'],process=False)
        vertices[obj['object_id']]=_array(mesh.vertices)*obj['scale']
    max_bottom_error=max_load_error=max_depth=max_illegal=0.;contact_count=0
    reference={};last={};pickup={key:[] for key in ids};bottoms={key:[] for key in ids};settled={}
    joints,legal=_robot_xml(plan,bindings)
    need(set(setup['legal_hand_rigid_owners'])==legal,'Legal hand/fixed-owner mask differs from independent source XML')
    robot={p for p in setup['contact_targets'] if p.startswith('/World/Robot/')}
    paths=setup['body_paths'];reverse={p:key for key,p in paths.items()}
    contacts_path=execution/'MULTI_OBJECT_CONTACTS.json';bindings.add(contacts_path,evaluation['contacts_sha256'])
    contact_iter=iter_contact_rows(contacts_path,bindings)
    for i,row in enumerate(rows):
        need(row.get('step')==i+1 and row.get('phase')==phases[i]
             and abs(row.get('time_s',-1)-(i+1)*DT)<=1e-12 and set(row['objects'])==set(ids),
             'Native all-body trace clock/phase/identity mismatch')
        contacts=next(contact_iter,None);need(isinstance(contacts,dict) and contacts.get('step')==i+1,'Exactly one full contact row per native tick required')
        loads={key:np.zeros(5) for key in ids};illegal=dict.fromkeys(ids,0.);total=dict.fromkeys(ids,0.)
        for group in ('object_contacts','robot_contacts'):
            need(isinstance(contacts.get(group),list),'Both complete contact streams required')
            for c in contacts[group]:
                pair=c['bodies'];need(isinstance(pair,list) and len(pair)==2 and all(isinstance(p,str) for p in pair),'Contact pair invalid')
                need(not any('Bin' in p for p in pair),'Bin geometry in lift contacts')
                _array(c['position_world_m'],(3,))
                # Original RobotContactReports does not export normals; the
                # all-object contact view does and its complete values stay.
                if group=='object_contacts':_array(c['raw_normal_world'],(3,))
                elif 'raw_normal_world' in c:_array(c['raw_normal_world'],(3,))
                force=float(c['normal_force_n']);separation=float(c['separation_m'])
                need(np.isfinite(force) and np.isfinite(separation),'Nonfinite native contact')
                max_depth=max(max_depth,-separation);contact_count+=1
                if group=='object_contacts':
                    need(pair[0] in reverse and pair[1] in setup['contact_targets'],'Unbound native object contact path')
                    if pair[1] in robot:
                        key=reverse[pair[0]];owner=pair[1].rsplit('/',1)[-1];magnitude=abs(force)
                        total[key]+=magnitude
                        for f,prefix in enumerate(('th','ff','mf','rf','lf')):
                            if owner.startswith(prefix):loads[key][f]+=magnitude
                        if owner not in legal:illegal[key]=max(illegal[key],magnitude)
        for key in ids:
            state=row['objects'][key];pose=_array(state['pose_xyzw'],(7,));_array(state['velocity'],(6,))
            need(abs(np.linalg.norm(pose[3:])-1)<1e-4,'Native quaternion not unit length')
            bottom=float((vertices[key]@Rotation.from_quat(pose[3:]).as_matrix()[2]+pose[2]).min())
            error=abs(bottom-state['source_bottom_m']);max_bottom_error=max(max_bottom_error,error)
            need(error<=1e-9,'Source mesh lowest vertex differs from reported bottom')
            fingers=_array(state['finger_loads_n'],(5,));error=float(np.max(np.abs(fingers-loads[key])));max_load_error=max(max_load_error,error)
            need(error<=1e-10 and abs(state['illegal_robot_support_force_n']-illegal[key])<=1e-10
                 and abs(state['robot_load_n']-total[key])<=1e-10,'All-body finger/illegal/total loads differ from actual contacts')
            max_illegal=max(max_illegal,illegal[key]);bottoms[key].append(bottom)
            if row['phase']=='settle':settled[key]=pose[:2].copy()
            if row['phase'] in ('home_to_pregrasp','approach'):
                need(key in settled and np.linalg.norm(pose[:2]-settled[key])<=.03,'Preclosure all-body push guard should have stopped execution')
            if row['phase']=='hold':reference[key]=pose[2]
            flag=bool(row['phase']=='lift' and key in reference and bottom>=.005
                      and pose[2]-reference[key]>=.005 and loads[key][0]>.001 and loads[key][1:].sum()>.001)
            pickup[key].append(flag);last[key]=(pose,bottom,loads[key])
    need(next(contact_iter,None) is None,'Extra native contact ticks')
    need(max_depth<=.001 and max_illegal<=.001,'All-body penetration/illegal arm support gate failed')
    need(abs(max_depth-evaluation['max_penetration_m'])<=1e-10
         and abs(max_illegal-evaluation['max_all_object_illegal_support_force_n'])<=1e-10,'Contact maxima mismatch')
    outcomes=evaluation.get('object_outcomes');need(isinstance(outcomes,list) and len(outcomes)==len(ids),'Complete per-object outcomes required')
    need(len({o['object_id'] for o in outcomes})==len(ids) and {o['object_id'] for o in outcomes}==set(ids),'Duplicate/missing/extra per-object lift identity')
    recomputed={};durations={}
    for outcome in outcomes:
        key=outcome['object_id'];pose,bottom,fingers=last[key]
        success=bool(pickup[key][-1] and pose[2]-reference[key]>=.05)
        need(type(outcome.get('task_success')) is bool and outcome['task_success']==success,'Per-body final lift label differs from physical trace')
        need(outcome.get('observed_final_lift_success') is success,'Observed per-body lift result mismatch')
        trailing=longest=run=0
        for flag in pickup[key]:run=run+1 if flag else 0;longest=max(longest,run)
        trailing=run
        for field,value in (('qualifying_steps',sum(pickup[key])),('longest_observed_pickup_run_s',longest*DT),
                            ('trailing_observed_pickup_run_s',trailing*DT),('final_reference_rise_m',pose[2]-reference[key]),
                            ('final_source_bottom_m',bottom)):
            need(abs(outcome[field]-value)<=1e-9,'Per-body source/rise/duration diagnostic mismatch: '+field)
        recomputed[key]=success;durations[key]=dict(longest_s=longest*DT,trailing_s=trailing*DT)
    need(type(evaluation.get('task_success')) is bool and evaluation['task_success']==any(recomputed.values()),'Any-original-body success mismatch')
    # Independently expand 24 commanded coordinates into the actual 28 joints.
    native_path=execution/'evaluation.json';bindings.add(native_path);native=read(native_path)
    native_trace=execution/'trace.npz';bindings.add(native_trace,native['trace_sha256'])
    need(native.get('complete') is True and native.get('total_physics_steps')==n
         and all(native.get('gates',{}).get(k) is True for k in GATES[:7]),'All seven original native gates required')
    for field in ('source_sha256','input_sha256'):
        if isinstance(native.get(field),dict):bindings.mapping(native[field])
    with np.load(native_trace,allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
    actual_names=list(map(str,actual['joint_names']));command_names=list(map(str,arrays['joint_names']))
    need(len(actual_names)==28 and len(set(actual_names))==28 and set(actual_names)==set(joints)
         and set(command_names)=={k for k,v in joints.items() if v['mimic'] is None},'Actual 28 coordinates do not match source URDF')
    expected=_full_target(command_names,arrays['joint_target'],joints,actual_names)
    need(actual['joint_target'].shape==expected.shape and np.array_equal(actual['joint_target'],expected),'Actual native command target differs from original 24+4 mimic expansion')
    need(np.array_equal(actual['phase'],arrays['phase']) and np.array_equal(actual['desired_palm'],arrays['desired_palm'])
         and np.array_equal(actual['time_s'],arrays['time_s']),'Actual desired palm/phase/clock changed')
    position=_array(actual['joint_pos'],(n,28));max_limit=max_mimic=0.
    for j,name in enumerate(actual_names):
        low,high=joints[name]['limits'];max_limit=max(max_limit,float(np.max(np.maximum(low-position[:,j],position[:,j]-high))))
        if joints[name]['mimic'] is not None:
            driver,mult,offset=joints[name]['mimic'];max_mimic=max(max_mimic,float(np.max(np.abs(position[:,j]-position[:,actual_names.index(driver)]*mult-offset))))
    need(max_limit<=.002 and max_mimic<=.002,'Measured native limit/mimic violation')
    primary=scene['target_object_id'];native_pose=_array(actual['object_pose_xyzw'],(n,7))
    need(np.max(np.abs(native_pose-np.asarray([r['objects'][primary]['pose_xyzw'] for r in rows])))<=1e-10,
         'Shared native primary body trace differs from all-body trace')
    return dict(physics_steps=n,dynamic_object_steps=n*len(ids),contact_points=contact_count,
        max_source_bottom_error_m=max_bottom_error,max_finger_load_error_n=max_load_error,
        max_penetration_m=max_depth,max_illegal_robot_support_force_n=max_illegal,
        max_joint_limit_violation_rad=max_limit,max_mimic_error_rad=max_mimic,
        actual_command_coordinates=28,independent_command_coordinates=24,
        object_success=recomputed,pickup_duration_diagnostics=durations,
        no_added_minimum_hold_duration=True,bin_geometry_count=0)


def audit_trial(prepared_dir,split_manifest=None):
    start=time.monotonic();directory=Path(prepared_dir).resolve();bindings=Bindings()
    result=dict(schema=SCHEMA,status='failed',formal_training_eligible=False,label=None,
        prepared_dir=str(directory),checks=[],costs=dict(Kit_starts=0,physics_steps=0,planning_calls=0,CGR_forwards=0,training_updates=0))
    try:
        bindings.add(__file__)
        p=directory/'LIFT_PIPELINE_PROTOCOL.json';bindings.add(p);protocol=read(p)
        need(protocol.get('task')==TASK and protocol.get('bin','missing') is None,'New lift protocol required')
        for key in ('inputs','sources','capture_source_bindings'):bindings.tree(protocol.get(key,{}))
        scene_path=Path(protocol['inputs']['scene']['path']);scene=read(scene_path)
        ids,original,converted,split=_identity(scene,split_manifest,bindings)
        result.update(original_object_ids=ids,all_scene_source_mesh_sha256=original,
            converted_scene_source_mesh_sha256=converted,split=split,scene_id=scene['scene_id'])
        need(protocol.get('original_object_ids')==ids and protocol.get('primary_is_candidate_target') is False
             and protocol.get('candidate_budget')==1 and protocol.get('fallback_candidate') is False,
             'Whole-scene identities/fixed first candidate protocol required')
        for path,digest in scene.get('source_sha256',{}).items():
            resolved=protocol.get('scene_archive_sources',{}).get(str(Path(path).resolve()),path);bindings.add(resolved,digest)
        bindings.mapping(scene.get('input_sha256',{}))
        result['checks'].append('all_scene_original_visual_identity_and_converted_geometry_split')
        prep_path=directory/'LIFT_PIPELINE_RESULTS.json';bindings.add(prep_path);preparation=read(prep_path)
        if not (directory/'READY.json').exists() or not (directory/'COLLECTION_RECORD.json').exists():
            result.update(status='not_executed' if not (directory/'EXECUTION_STARTED.json').exists() else 'incomplete',
                reason='No complete persisted actual collection record; never a negative training label')
        else:
            ready_path=directory/'READY.json';bindings.add(ready_path,preparation['ready_sha256']);ready=read(ready_path)
            bindings.tree(ready)
            need(preparation.get('status')=='ready' and ready.get('task')==TASK
                 and ready['scene']==protocol['inputs']['scene'],'Actual ready scene/protocol mismatch')
            plan=read(ready['plan']['path']);candidate=read(ready['candidate']['path']);command_binding=read(ready['command_binding']['path'])
            for field in ('source_sha256','input_sha256','extra_input_sha256'):
                bindings.mapping(plan.get(field,{}))
            arrays,names,phases,counts=_commands(directory,ready,plan,candidate,command_binding,bindings)
            need(command_binding['execution_spec'].get('original_object_ids')==ids,'Command must bind all original scene identities')
            record_path=directory/'COLLECTION_RECORD.json';bindings.add(record_path);record=read(record_path)
            bindings.tree(record.get('evaluation_binding',{}));evaluation=read(record['evaluation_binding']['path'])
            need(record.get('candidate')==candidate and record.get('command_binding')==command_binding
                 and record.get('task')==TASK and evaluation.get('task')==TASK,'Persisted record differs from candidate/action/task')
            need(record.get('formal_training_eligible') is True and type(record.get('label')) is int and record['label'] in (0,1),
                 'Only actually eligible persisted binary labels can be certified')
            need(evaluation.get('complete') is True and evaluation.get('label_eligible') is True
                 and not evaluation.get('error') and not evaluation.get('physical_stop')
                 and all(evaluation.get('gates',{}).get(name) is True for name in GATES),
                 'Incomplete/infrastructure/invalid execution cannot become label0')
            need(evaluation.get('collection_candidate_sha256')==candidate['candidate_sha256']
                 and evaluation.get('commands_sha256')==ready['commands']['sha256']
                 and evaluation.get('scene_sha256')==ready['scene']['sha256']
                 and evaluation.get('original_object_ids')==ids and evaluation.get('completed_phases')==PHASES
                 and evaluation.get('legacy_primary_task_success_used') is False
                 and evaluation.get('state_writes_during_execution')==0 and evaluation.get('bin_geometry_count')==0,
                 'Evaluation identity/command/no-reset/bin-free semantics mismatch')
            execution=Path(record['evaluation_binding']['path']).parent
            need(execution.resolve()==(directory/'execution').resolve(),'Record points outside actual trial execution')
            started_path=directory/'EXECUTION_STARTED.json';bindings.add(started_path);started=read(started_path)
            need(started.get('task')==TASK and started.get('ready_sha256')==sha(ready_path),'Once-only execution start binding missing')
            receipt_path=directory/'LIFT_PIPELINE_EXECUTION.json';bindings.add(receipt_path);receipt=read(receipt_path)
            need(receipt.get('complete') is True and receipt.get('new_labels')==1 and receipt.get('label')==record['label'],
                 'Pipeline final receipt did not complete this eligible label')
            streamed_path=directory/'STREAMED_EXECUTION_ENTRY.json'
            if streamed_path.exists():
                bindings.add(streamed_path);entry=read(streamed_path)
                need(entry.get('schema')=='anydex_boya_streamed_lift_execution_entry_v1'
                     and entry.get('status')=='finalized' and entry.get('changed_sources')==[]
                     and entry.get('renderer_active_gpu')==0 and entry.get('renderer_multi_gpu_enabled') is False
                     and entry.get('physics_device')=='cpu' and entry.get('contact_records_omitted')==0,
                     'Lossless execution wrapper did not finalize its GPU0/source receipt')
                bindings.mapping(entry['source_sha256']);bindings.tree(entry['pipeline_execution_receipt'])
                need(entry['pipeline_execution_receipt']==dict(path=str(receipt_path),sha256=sha(receipt_path)),
                     'Wrapper final receipt does not bind the completed pipeline receipt')
            result['checks'].append('source_candidate_ready_actual_commands_record_and_execution_bindings')
            numeric=audit_runtime(scene,plan,arrays,execution,evaluation,bindings)
            need(record['label']==int(any(numeric['object_success'].values())),'Persisted binary label differs from independent all-body numeric audit')
            result.update(status='pass',formal_training_eligible=True,label=record['label'],runtime=numeric,
                candidate_sha256=candidate['candidate_sha256'],commands_sha256=ready['commands']['sha256'],phase_counts=counts)
            result['checks'].extend(['all_contacts_lossless_decode_and_all_body_force_recomputation',
                'all_source_vertex_bottoms_and_final_lift_labels','actual_28_targets_from_24_source_coordinates_and_mimics'])
        bindings.verify()
    except BaseException as error:
        result.update(status='failed',formal_training_eligible=False,label=None,error=repr(error))
    result.update(audit_inputs_sha256=bindings.files,audit_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),wall_s=time.monotonic()-start)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--split-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    need(not a.output.exists(),'Fresh audit output path required')
    result=audit_trial(a.prepared_dir,a.split_manifest);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],label=result['label'],formal_training_eligible=result['formal_training_eligible'])))


if __name__=='__main__':main()
