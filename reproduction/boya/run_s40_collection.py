"""Frozen S40 Boya collection: all40 train meshes, four types and four depths.

A new version preserves the two-object pilot and its physical records. Ordinary
empty geometry and certified task stops are null; unknown errors stop dispatch.
No DGN decision head, test object or learned-score threshold enters collection.
"""
from __future__ import annotations
import argparse
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
BOYA = ROOT/'reproduction/boya'
if str(BOYA) not in sys.path:
    sys.path.insert(0, str(BOYA))
from run_collection_pipeline_v2 import (Pipeline as OriginalPipeline, bounded_jobs,
    sha, read, save, immutable, hashes, verify, env, native_profile_identity)
from render_s40_observations import S40_IDS
NATIVE = Path('/home/wangrenpeng/CanonDex/scripts/research/preprocess_native_object.py')
GESTURES = [1,6,7,8]
DEPTHS = [0,1,2,3]


def job_id(oid, gesture, depth):
    return f'{oid}_g{gesture}_d{depth}'


def validate_null(doc, job, calibration):
    """Only a source-bound zero-step ordinary geometry null may absorb nonzero."""
    obs = read(job['observation_json'])
    verify({job['source_npz']:doc['source_candidates_sha256'],
            job['observation_json']:doc['source_observation_sha256'],
            str(calibration):doc['calibration_sha256']})
    if (doc.get('status') != 'no_hand_geometry_in_declared_budget'
            or doc.get('physics_steps') != 0 or doc.get('training_label') is not None
            or doc.get('gesture_id') != job['gesture_id']
            or obs.get('object_id') != job['object_id']
            or not isinstance(doc.get('records'), list)):
        raise ValueError('Foreign, physical or labelled geometry null')
    depths = {x['additional_depth_index'] for x in doc['records'] if 'additional_depth_index' in x}
    if depths and depths != {job['extra_depth_index']}:
        raise ValueError('Geometry null belongs to another declared insertion')
    return dict(trial_id=job['trial_id'], object_id=job['object_id'],
                gesture_id=job['gesture_id'],extra_depth_index=job['extra_depth_index'],
                status='no_candidate',training_label=None,physics_steps=0)


def validate_scene(manifest):
    rows = manifest.get('objects', [])
    if (manifest.get('schema') != 'anydex_boya_s40_scene_manifest_v1'
        or manifest.get('scale') != 1. or len(rows) != 40
        or {r['graspnet_object_id'] for r in rows} != S40_IDS
        or len({r['object_id'] for r in rows}) != 40
        or any(r['split'] != 'train' or r['scale'] != 1. for r in rows)):
        raise ValueError('Complete original S40 source-ID union, all train, scale1 required')
    verify({manifest['source_inventory']:manifest['source_inventory_sha256']})
    return rows


def verify_frame_binding(row, receipt):
    obs=read(row['observation_json']);binding=receipt.get('scene_binding',{})
    if (receipt.get('object_id')!=row['object_id'] or receipt.get('full_representation_forward')is not True
        or Path(binding.get('observation_json','')).resolve()!=Path(row['observation_json']).resolve()
        or binding.get('observation_sha256')!=sha(row['observation_json'])
        or binding.get('T_WC')!=obs['T_WC'] or binding.get('object_pose_as_network_input')is not False):
        raise ValueError('CGR frame does not bind current calibrated observation')
    expected={str(Path(item['path']).resolve()):item['sha256']for item in obs['files'].values()}
    provided={str(Path(path).resolve()):item['sha256']for path,item in receipt.get('input_files',{}).items()}
    if expected!=provided:raise ValueError('CGR input hashes differ from declared RGB-D')
    verify(expected)


def inherit_pilot(pilot, objects, calibration, backend_config):
    """Retain exactly eight previously terminal slots; never replay five grasps."""
    protocol = read(pilot/'PILOT_PROTOCOL.json')
    progress = read(pilot/'PROGRESS.json')
    if (progress.get('status') != 'complete' or protocol['object_ids'] != ['graspnet_000','graspnet_002']
            or protocol['gesture_ids'] != GESTURES or protocol['extra_depth_indices'] != [0]
            or protocol['slots'] != 8):
        raise ValueError('Completed original two-object pilot required')
    verify(protocol['input_sha256']); verify(protocol['source_sha256'])
    if (protocol['input_sha256'].get(str(calibration)) != sha(calibration)
            or protocol['input_sha256'].get(str(backend_config)) != sha(backend_config)):
        raise ValueError('Pilot calibration or backend differs')
    old_scene = read(pilot/'scenes/objects.json')
    byid = {r['object_id']:r for r in old_scene['objects']}
    for row in objects:
        old = byid[row['object_id']]
        for key in ['mesh_sha256','urdf_sha256','pose_sha256','world_T_object','scale','object_mass_kg','object_friction']:
            if row[key] != old[key]:
                raise ValueError('Pilot source geometry or placement changed')
    queue = read(pilot/'planning/queue_manifest.json')
    if not queue['complete'] or queue['errors'] or len(queue['trials']) != 5 or len(queue['nulls']) != 3:
        raise ValueError('Original pilot slot partition changed')
    actual = []
    records = []
    refs = [pilot/'PILOT_PROTOCOL.json',pilot/'PROGRESS.json',pilot/'planning/queue_manifest.json',
            pilot/'scenes/objects.json']
    for oid in protocol['object_ids']:
        result_path = pilot/'execution'/oid/'RESULTS.json'
        result = read(result_path); refs += [result_path,pilot/'execution'/oid/'PREPARED.json']
        if result.get('infrastructure_error') or result['status'] not in ['complete','complete_with_null']:
            raise ValueError('Pilot physical closeout is not complete')
        for row in result['trials']:
            if row.get('label_eligible') is not True or row.get('label') not in (0,1) or row['physics_steps'] != 2100:
                raise ValueError('Pilot eligible execution record changed')
            actual.append(row)
            rp = pilot/'execution'/oid/'trials'/row['output_identity']/'training_record.json'
            records.append(read(rp));refs += [rp,Path(row['execution_dir'])/'evaluation.json',Path(row['execution_dir'])/'trace.npz']
    if len(actual) != 5 or len(records) != 5 or sum(r['physics_steps'] for r in actual) != 10500:
        raise ValueError('Original five actual trials must be retained exactly once')
    ids = {r['trial_id'] for r in queue['trials']+queue['nulls']}
    expected = {job_id(oid,g,0) for oid in protocol['object_ids'] for g in GESTURES}
    if ids != expected:
        raise ValueError('Original eight terminal slots do not reconcile')
    return dict(pilot=str(pilot),terminal_slot_ids=sorted(ids),
                queue=queue,actual_rows=actual,training_records=records,
                frozen_refs=hashes(refs),physical_steps=10500,
                inherited_original_protocol_sha256=sha(pilot/'PILOT_PROTOCOL.json'))


def freeze(args):
    run = args.run_dir.resolve()
    protocol_path = run/'protocol.json'
    if protocol_path.exists():
        p = read(protocol_path)
        if p.get('schema') != 'anydex_boya_s40_collection_v1':
            raise ValueError('Foreign collection protocol')
        for key in ['scene_manifest','calibration','backend_config','checkpoint','native_deps','pilot']:
            if str(getattr(args,key).resolve()) != p['config'][key]:
                raise ValueError('Frozen option changed: '+key)
        for key in ['geometry_workers','plan_workers','execution_workers','native_workers','device']:
            alias = 'representation_device' if key=='device' else key
            if getattr(args,key) != p['config'][alias]:
                raise ValueError('Frozen concurrency/device changed: '+key)
        verify(p['input_sha256']);verify(p['source_sha256']);verify(p['manifest_sha256'])
        return p
    if run.exists() and any(run.iterdir()):
        raise ValueError('New collection requires an empty directory')
    scene = read(args.scene_manifest);objects = validate_scene(scene)
    inherited = inherit_pilot(args.pilot.resolve(),objects,args.calibration.resolve(),args.backend_config.resolve())
    run.mkdir(parents=True,exist_ok=True)
    cfg = dict(scene_manifest=str(args.scene_manifest.resolve()),calibration=str(args.calibration.resolve()),
        backend_config=str(args.backend_config.resolve()),checkpoint=str(args.checkpoint.resolve()),
        native_deps=str(args.native_deps.resolve()),pilot=str(args.pilot.resolve()),
        representation_device=args.device,geometry_workers=args.geometry_workers,
        plan_workers=args.plan_workers,execution_workers=args.execution_workers,native_workers=args.native_workers,
        physics_device='cpu',execution_mode='serial',batch_size=16,candidate_limit=500,keep=8,max_arm_plans=4,seed=20261107,
        gesture_ids=GESTURES)
    rebased = copy.deepcopy(scene)
    rebased.update(source_scene_manifest=str(args.scene_manifest.resolve()),source_scene_manifest_sha256=sha(args.scene_manifest))
    for row in rebased['objects']:
        oid = row['object_id'];old = oid in ['graspnet_000','graspnet_002']
        row['observation_json'] = str(args.pilot.resolve()/'observations'/oid/'observation.json') if old else str(run/'observations'/oid/'observation.json')
        row['representation_dir'] = str(args.pilot.resolve()/'representations'/oid) if old else str(run/'representations'/oid)
        row['observation'] = row['observation_json'];row['observation_inherited_from_pilot'] = old
    save(run/'objects.json',rebased)
    save(run/'INHERITED_PILOT.json',inherited)
    fresh = [r for r in rebased['objects'] if not r['observation_inherited_from_pilot']]
    save(run/'representation_manifest.json',dict(objects=[{k:r[k] for k in ['object_id','split','observation_json','representation_dir']} for r in fresh]))
    slots = [dict(trial_id=job_id(r['object_id'],g,d),object_id=r['object_id'],split='train',gesture_id=g,extra_depth_index=d,
        candidate_dir=str(run/'pools'/r['object_id']/f'g{g}'/f'd{d}'),observation_json=r['observation_json'],
        source_npz=str(Path(r['representation_dir'])/'raw_camera_candidates.npz'),calibration=cfg['calibration'],max_arm_plans=4)
        for r in rebased['objects'] for g in GESTURES for d in DEPTHS]
    files = {}
    def add(path, expected=None):
        path = str(Path(path).resolve());actual=sha(path)
        if expected is not None and actual != expected:raise ValueError('Source input changed: '+path)
        files[path] = actual
    for name in ['scene_manifest','calibration','backend_config','checkpoint']:
        add(cfg[name])
    add(scene['source_inventory'],scene['source_inventory_sha256'])
    files.update(inherited['frozen_refs'])
    cal=read(cfg['calibration']);backend=read(cfg['backend_config'])
    for value in backend.values():
        if isinstance(value,str) and value.startswith('/') and Path(value).is_file():add(value)
    for path in Path(backend['reference_root']).rglob('*.json'):add(path)
    for path,item in cal['source_lock']['source_files'].items():add(path,item['sha256'])
    for path,digest in cal['source_lock']['readonly_fk_dependencies'].items():add(path,digest)
    for row in rebased['objects']:
        for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')]:add(row[pk],row[hk])
        for part in row['convex_parts']:add(part['path'],part['sha256'])
        for path,digest in row['source_input_sha256'].items():add(path,digest)
        if row['observation_inherited_from_pilot']:
            obs=read(row['observation_json']);add(row['observation_json'])
            for item in obs['files'].values():add(item['path'],item['sha256'])
            receipt=read(Path(row['representation_dir'])/'RESULTS.json');add(Path(row['representation_dir'])/'RESULTS.json')
            for item in receipt['output_files'].values():add(item['path'],item['sha256'])
    source_paths=[*BOYA.glob('*.py'),NATIVE,ROOT/'robot_dh3.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'reproduction/runtime/run_official_rgbd_frame.py']
    canon=NATIVE.parents[2];source_paths += [*(canon/'canondex').rglob('*.py'),*(canon/'scripts/research').glob('*.py')]
    group_order=sorted(rebased['objects'],key=lambda r:hashlib.sha256(('s40-internal-20261107:'+r['source_visual_mesh_sha256']).encode()).hexdigest())
    internal_groups={name:[dict(object_id=r['object_id'],mesh_sha256=r['source_visual_mesh_sha256'])for r in rows]for name,rows in [('fit',group_order[:32]),('internal_check',group_order[32:])]}
    protocol=dict(schema='anydex_boya_s40_collection_v1',internal_mesh_groups=internal_groups,config=cfg,slots=slots,slot_count=640,split_slot_counts={'train':640,'validation':0},
        gesture_ids=GESTURES,extra_depth_indices=DEPTHS,per_object_slot_limit=16,valid_label_targets_per_gesture={'train':100,'validation':0},
        valid_label_target_totals={'train':400,'validation':0},selection_sha256=sha(run/'objects.json'),
        input_sha256=files,source_sha256=hashes(source_paths),manifest_sha256=hashes([run/'objects.json',run/'representation_manifest.json',run/'INHERITED_PILOT.json']),
        inherited_slot_count=8,new_slot_count=632,inherited_actual_trials=5,inherited_physics_steps=10500,
        inherited_protocol_sha256=inherited['inherited_original_protocol_sha256'],
        target_boundary='100 usable positive/negative labels per gesture is a target, not guaranteed. All640 frozen slots retained including null/deficits; no result-conditioned replacement, extra grasp or fake negative.',
        scope='All40 GraspNet-derived S40 train identities at scale1, one native-axis tabletop placement per object, fixed four-depth coverage Boya simulation port, both flips and existing nominal closure/arm gates. Not real hardware or full paper success rate reproduction.',
        no_test_objects=True,no_decision_head_loaded=True,new_training_updates=0,created_unix_s=time.time())
    verify(files);save(protocol_path,protocol)
    return protocol


class Pipeline(OriginalPipeline):
    def __init__(self,run,protocol):
        super().__init__(run,protocol)
        self.inherited=read(run/'INHERITED_PILOT.json')
        self.inherited_ids=set(self.inherited['terminal_slot_ids'])
        self.progress.update(inherited_actual_grasps=5,inherited_physics_steps=10500,new_physical_steps=0,physical_steps=10500)

    def publish(self,status):
        super().publish(status)
        print('S40_COLLECTION '+json.dumps(dict(status=status,time_unix=time.time(),details=self.progress.get('detail'))),flush=True)

    def native(self):
        output=self.run/'native_preprocessing';output.mkdir(exist_ok=True)
        physics=read(self.c['backend_config'])['physics_config'];profile=native_profile_identity(physics)
        pilot=Path(self.c['pilot'])
        def one(row):
            oid=row['object_id'];old=pilot/'native_assets'/oid/'result.json';target=output/oid
            resultfile=old if oid in ['graspnet_000','graspnet_002'] else target/'result.json'
            if not resultfile.exists():
                if target.exists():raise ValueError('Partial native preparation retained')
                self.command('native_'+oid,[sys.executable,'-B',str(NATIVE),'--object-root',row['object_root'],'--scale','1',
                    '--physics-config',physics,'--cache-dir',str(self.run/'native_object_cache'),'--output-dir',str(target),'--max-failed-partitions','32'])
            doc=read(resultfile);bound=doc['binding']['inputs']
            if not(doc.get('complete') and doc.get('trial_physics_steps')==0 and doc.get('original_or_refined_within_20um')):
                raise ValueError('Native asset gate not passed: '+oid)
            if (bound['object_id']!=oid or bound['scale']!=1. or bound['physics_profile_sha256']!=profile
                or bound['max_failed_partitions']!=32 or bound['tolerance_m']!=20e-6):raise ValueError('Foreign native asset binding')
            verify({str(Path(row['object_root'])/k):v for k,v in bound['source_files'].items()})
            verify({str(NATIVE.parents[2]/k):v for k,v in bound['implementations'].items()})
            cache=Path(doc['cache_directory']);assets=[resultfile]
            for name,digest in doc.get('artifact_sha256',{}).items():verify({str(cache/name):digest});assets.append(cache/name)
            repair=doc.get('repair_path')
            if repair:assets.append(Path(repair))
            return dict(object_id=oid,result_json=str(resultfile),inherited=old==resultfile,
                repair_path=repair,repair_sha256=sha(repair)if repair else None,artifacts=list(map(str,assets)))
        def completed(row,value,bad):
            self.progress['detail']=dict(native_finished=self.progress.get('detail',{}).get('native_finished',0)+1,total=40,object_id=row['object_id'],error=bad)
            self.publish('native_assets_running')
        work=bounded_jobs(self.objects,self.c['native_workers'],one,completed)
        if any(r['failed'] for r in work):raise RuntimeError('Native source import error; inflight drained')
        result=[r['value']for r in work];path=output/'SUMMARY.json'
        save(path,dict(complete=True,results=result,new_physics_steps=0,physics_profile_sha256=profile))
        return dict(objects=40,new_objects=38,adopted_objects=2),[path,*[p for row in result for p in row['artifacts']]]

    def observations(self):
        fresh=[r for r in self.objects if not r['observation_inherited_from_pilot']]
        output=self.run/'observations'
        if output.exists():raise ValueError('Partial observations retained; no automatic re-render')
        self.command('render',[sys.executable,'-B',str(BOYA/'render_s40_collection_observations.py'),
            '--objects-json',str(self.run/'objects.json'),'--output-dir',str(output),'--object-ids',*[r['object_id']for r in fresh]])
        index=read(output/'OBSERVATIONS.json');assets=[output/'OBSERVATIONS.json']
        if index['status']!='complete' or len(index['observations'])!=38 or index['source_objects_sha256']!=sha(self.run/'objects.json'):
            raise ValueError('Foreign/incomplete S40 RGB-D index')
        for row in self.objects:
            path=Path(row['observation_json']);doc=read(path);assets.append(path)
            if (doc['object_id']!=row['object_id'] or doc['object_scale']!=1. or doc['world_T_object']!=row['world_T_object']
                or doc.get('explicit_physics_steps')!=0 or doc.get('robot_present')):raise ValueError('Observation geometry or scene identity changed')
            for item in doc['files'].values():verify({item['path']:item['sha256']});assets.append(Path(item['path']))
        return dict(objects=40,inherited=2,new_renders=38,new_physics_steps=0),assets

    def representation(self):
        output=self.run/'representation_stage'
        self.command('representation',[sys.executable,'-B',str(BOYA/'run_collection_representation.py'),
            '--manifest',str(self.run/'representation_manifest.json'),'--run-dir',str(output),
            '--device',self.c['representation_device'],'--checkpoint',self.c['checkpoint'],
            '--native-deps',self.c['native_deps'],'--seed',str(self.c['seed'])])
        batch=output/'BATCH_REPRESENTATION.json';doc=read(batch)
        if (doc.get('manifest_sha256')!=sha(self.run/'representation_manifest.json') or doc['status']!='all_representation_forwards_complete'
            or doc['network_forward_count']!=38 or doc['model_load_count']!=1 or not doc['manifest_unchanged_at_end']):
            raise ValueError('Expected38 actual CGR forwards and one model load')
        assets=[batch]
        for row in self.objects:
            path=Path(row['representation_dir'])/'RESULTS.json';receipt=read(path);assets.append(path)
            verify_frame_binding(row,receipt)
            for item in receipt['output_files'].values():verify({item['path']:item['sha256']});assets.append(Path(item['path']))
        return dict(objects=40,model_load_count=1,new_network_forwards=38,inherited_network_forwards=2),assets

    def geometry(self):
        def one(job):
            output=Path(job['candidate_dir']);generation=output/'generation.json';null=output/'no_candidate.json'
            if output.exists():raise ValueError('Partial pool preserved; no automatic retry')
            row=next(r for r in self.objects if r['object_id']==job['object_id'])
            argv=[sys.executable,'-B',str(BOYA/'build_collection_pool.py'),'--observation',job['observation_json'],
                '--frame-dir',row['representation_dir'],'--calibration',self.c['calibration'],'--gesture-id',str(job['gesture_id']),
                '--extra-depth-index',str(job['extra_depth_index']),'--candidate-limit','500','--keep','8','--width-expansion','1.7',
                '--opponent','ff','--output-dir',str(output)]
            proc=self.command('pool_'+job['trial_id'],argv,allowed_failure=True)
            if generation.exists():
                if proc['returncode']!=0 or null.exists():raise RuntimeError('Conflicting or failed generated pool')
                d=read(generation);verify(d['input_sha256']);verify({str(output/'candidates.npz'):d['candidates_sha256']})
                if (not d['complete'] or d['gesture_id']!=job['gesture_id'] or d['additional_depth_index']!=job['extra_depth_index']
                    or Path(d['source_observation'])!=Path(job['observation_json']) or Path(d['source_frame'])!=Path(row['representation_dir'])):
                    raise ValueError('Pool source/slot identity differs')
                return dict(trial_id=job['trial_id'],status='ready'),[generation,output/'candidates.npz',output/'world_object_pose.json']
            if null.exists():
                result=validate_null(read(null),job,self.c['calibration'])
                binding=output/'NULL_BINDING.json';save(binding,dict(**result,geometry_command=proc,
                    no_candidate_sha256=sha(null),declared_source_sha256={job['source_npz']:sha(job['source_npz']),job['observation_json']:sha(job['observation_json'])},
                    extra_depth_binding='Executed argv --extra-depth-index and source-bound child no_candidate receipt; no physical execution'))
                return result,[null,binding]
            raise RuntimeError('Geometry failed without normal source-bound null')
        pending=[j for j in self.p['slots'] if j['trial_id'] not in self.inherited_ids]
        done=[]
        def completed(job,value,bad):
            done.append((job,value,bad));self.progress['detail']=dict(geometry_finished=len(done),new_total=632,inherited=8,last=job['trial_id'],error=bad)
            self.publish('geometry_running')
        work=bounded_jobs(pending,self.c['geometry_workers'],one,completed)
        if any(r['failed'] for r in work):raise RuntimeError('Geometry infrastructure failure; no further dispatch')
        values=[r['value']for r in work];path=self.run/'POOL_PREPARATION.json'
        save(path,dict(complete=True,jobs=[v for v,_ in values],inherited_terminal_slots=self.inherited['terminal_slot_ids'],new_physics_steps=0,new_network_forwards=0))
        return dict(new_slots=632,inherited_slots=8,ready=sum(v['status']=='ready'for v,_ in values),no_candidate=sum(v['status']=='no_candidate'for v,_ in values)),[path,*[p for _,pp in values for p in pp]]

    def planning(self):
        from prepare_collection_queue import prepare_one
        native=read(self.run/'native_preprocessing/SUMMARY.json');repairs={r['object_id']:r for r in native['results']}
        jobs=copy.deepcopy([j for j in self.p['slots'] if j['trial_id']not in self.inherited_ids])
        for job in jobs:
            r=repairs[job['object_id']]
            if r['repair_path']:job.update(collision_refinement=r['repair_path'],collision_refinement_sha256=r['repair_sha256'])
        path=self.run/'preparation_jobs.json';save(path,dict(backend_config=self.c['backend_config'],jobs=jobs))
        output=self.run/'planning';output.mkdir(exist_ok=False);manifest=output/'queue_manifest.json'
        old=self.inherited['queue']
        doc=dict(schema='anydex_boya_collection_queue_v1',complete=False,trials=copy.deepcopy(old['trials']),nulls=copy.deepcopy(old['nulls']),errors=[],
            jobs_json=str(path),jobs_sha256=sha(path),source_sha256=hashes([BOYA/n for n in ['prepare_collection_queue.py','trial_record.py','asset_adapter.py','native_arm.py']]),
            physical_steps=0,learned_Boya_head_loaded=False,inherited_trial_ids=sorted(self.inherited_ids))
        save(manifest,doc)
        def one(job):
            verify(self.p['source_sha256']);value=prepare_one(job,output,Path(self.c['backend_config']));verify(self.p['source_sha256'])
            if value['status']not in ['ready','no_hand_geometry_in_declared_budget','no_full_arm_plan_in_declared_budget']:
                raise ValueError('Unknown arm preparation outcome')
            return dict(value,gesture_id=job['gesture_id'],extra_depth_index=job['extra_depth_index'],split='train')
        def completed(job,value,bad):
            if bad:doc['errors'].append(dict(trial_id=job['trial_id'],error=value.get('error')))
            else:doc['trials'if value['status']=='ready'else'nulls'].append(value)
            save(manifest,doc);self.progress['detail']=dict(terminal_slots=len(doc['trials'])+len(doc['nulls']),total=640,arm_ready=len(doc['trials']),nulls=len(doc['nulls']),last=job['trial_id'],error=bad)
            self.publish('planning_running')
        work=bounded_jobs(jobs,self.c['plan_workers'],one,completed)
        order={j['trial_id']:i for i,j in enumerate(self.p['slots'])}
        for key in ['trials','nulls']:doc[key].sort(key=lambda r:order[r['trial_id']])
        seen=[r['trial_id']for r in doc['trials']+doc['nulls']]
        doc.update(complete=not doc['errors']and len(seen)==640 and len(set(seen))==640,inputs_unchanged=sha(path)==doc['jobs_sha256'],
            not_dispatched_trial_ids=[j['trial_id']for j in self.p['slots']if j['trial_id']not in seen])
        save(manifest,doc)
        if not doc['complete']:raise RuntimeError('Arm planning infrastructure error; retain partial queue')
        assets=[path,manifest]
        for row in doc['trials']:
            verify(row['input_sha256']);plan=Path(row['plan_dir']);verify({str(plan/'commands.npz'):row['commands_sha256'],row['binding_json']:row['binding_sha256']})
            assets += [plan/n for n in ['plan.json','commands.npz','path_geometry.json','lift_arm_geometry.json']]
            assets.append(Path(row['binding_json']))
        for row in doc['nulls']:
            base=Path(self.c['pilot'])/'planning' if row['trial_id']in self.inherited_ids else output
            assets.append(base/row['trial_id']/'preparation.json')
        return dict(manifest=str(manifest),admitted=len(doc['trials']),nulls=len(doc['nulls']),inherited_actual=5,new_physics_steps=0),assets

    def execution(self):
        from s40_execution_outcomes import inspect_execution
        queue=read(self.run/'planning/queue_manifest.json');output=self.run/'execution'
        output.mkdir(exist_ok=False);(output/'object_queues').mkdir();(output/'terminal_certificates').mkdir()
        started=time.monotonic();jobs=[];certificates={};assets=[]
        progress=dict(schema='anydex_boya_s40_execution_v1',status='running',objects={},dispatch_events=[],peak_inflight_object_jobs=0,
            inherited_actual_grasps=5,inherited_physics_steps=10500)
        for obj in self.objects:
            oid=obj['object_id'];trials=[r for r in queue['trials']if r['object_id']==oid and r['trial_id']not in self.inherited_ids]
            nulls=[r for r in queue['nulls']if r['object_id']==oid and r['trial_id']not in self.inherited_ids]
            sub=dict(queue,trials=trials,nulls=nulls,parent_queue_manifest=str(self.run/'planning/queue_manifest.json'),
                parent_queue_manifest_sha256=sha(self.run/'planning/queue_manifest.json'))
            path=output/'object_queues'/(oid+'.json');save(path,sub);assets.append(path)
            jobs.append(dict(object_id=oid,manifest=str(path),ready=len(trials)))
        save(output/'PROGRESS.json',progress)
        def one(job):
            oid=job['object_id'];sub=read(job['manifest']);target=output/oid
            if not job['ready']:return dict(object_id=oid,status='no_ready_plans',result=dict(trials=[]),terminal_certificates={})
            save(output/'object_queues'/(oid+'.SUBMITTING.json'),dict(object_id=oid,trial_ids=[r['trial_id']for r in sub['trials']],manifest=job['manifest'],manifest_sha256=sha(job['manifest']),time_unix=time.time(),state='submission_intent_not_proof_of_unexecuted'))
            proc=self.command('execution_'+oid,[sys.executable,'-B',str(BOYA/'execute_collection_queue_v2.py'),
                '--manifest',job['manifest'],'--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size','16'],allowed_failure=True)
            doc=read(target/'RESULTS.json')if(target/'RESULTS.json').exists()else{}
            checked=inspect_execution(doc,[r['trial_id']for r in sub['trials']],proc['returncode'])
            return dict(object_id=oid,status='complete',output_dir=str(target),result=doc,terminal_certificates=checked['terminal_certificates'])
        def completed(job,value,bad):
            progress['objects'][job['object_id']]=dict(value,infrastructure_error=bad)
            for tid,cert in value.get('terminal_certificates',{}).items():
                certificates[tid]=cert;save(output/'terminal_certificates'/(tid+'.json'),cert)
            progress['wall_s']=time.monotonic()-started
            save(output/'PROGRESS.json',progress)
            vals=[r for v in progress['objects'].values()for r in v.get('result',{}).get('trials',[])]
            self.progress['detail']=dict(objects_finished=len(progress['objects']),objects=40,inherited_actual=5,
                new_actual=sum(r.get('physics_steps',0)>0 for r in vals),new_physics_steps=sum(r.get('physics_steps',0)for r in vals),
                eligible=5+sum(r.get('label_eligible')is True for r in vals),terminal_nulls=len(certificates),last=job['object_id'],error=bad)
            self.publish('execution_running')
        def dispatched(job,active):
            progress['peak_inflight_object_jobs']=max(progress['peak_inflight_object_jobs'],sum(bool(j['ready'])for j in active))
            progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],elapsed_s=time.monotonic()-started));save(output/'PROGRESS.json',progress)
        work=bounded_jobs(jobs,self.c['execution_workers'],one,completed,dispatched)
        failed=any(r['failed']for r in work)
        # Read every dispatched executor's durable result even if validation failed.
        for job in jobs:
            oid=job['object_id'];item=progress['objects'].setdefault(oid,dict(status='not_dispatched',result={}))
            rp=output/oid/'RESULTS.json'
            if rp.exists():item['result']=read(rp);item['output_dir']=str(output/oid)
        rows=list(self.inherited['actual_rows'])+[r for value in progress['objects'].values()for r in value.get('result',{}).get('trials',[])]
        order={j['trial_id']:i for i,j in enumerate(self.p['slots'])};byid={j['trial_id']:j for j in self.p['slots']}
        if len({r['trial_id']for r in rows})!=len(rows):raise ValueError('Duplicate physical trial identity')
        rows.sort(key=lambda r:order[r['trial_id']]);records=[]
        def add_record(value,tid,inherited=False):
            if value.get('label_eligible')is True and value.get('result')in (0,1):
                row=next(r for r in rows if r['trial_id']==tid)
                if (type(value['result'])is not int or value['result']!=row['label']
                    or value['binding']['trial_binding_sha256']!=row['trial_binding_sha256']
                    or value['total_physics_steps']!=row['physics_steps']):
                    raise ValueError('Training record differs from actual trial binding/outcome/count')
                artifact=value['execution_artifacts']
                verify({artifact['evaluation_json']:artifact['evaluation_sha256'],artifact['commands_npz']:artifact['commands_sha256']})
                if artifact['commands_sha256']!=row['commands_sha256']:
                    raise ValueError('Training record command bytes differ from actual trace')
                from trial_record import collect_record
                expected=collect_record(value['binding'],artifact['evaluation_json'],artifact['commands_npz'])
                if any(value.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Training export differs from original artifact-derived record')
                slot=byid[tid];value=copy.deepcopy(value)
                if (value['binding']['multifinger_pose_finger_type']!=slot['gesture_id']
                    or value['binding']['multifinger_pose_depth_type']!=slot['extra_depth_index']):
                    raise ValueError('Training record finger type/depth differs from declared slot')
                value['collection_context']=dict(trial_id=tid,object_id=slot['object_id'],split='train',gesture_id=slot['gesture_id'],
                    extra_depth_index=slot['extra_depth_index'],benchmark_only=False,protocol_sha256=sha(self.run/'protocol.json'),
                    inherited_from_pilot=inherited,original_protocol_sha256=self.inherited['inherited_original_protocol_sha256']if inherited else None)
                records.append(value)
        for row,value in zip(self.inherited['actual_rows'],self.inherited['training_records']):add_record(value,row['trial_id'],True)
        for item in progress['objects'].values():
            if not item.get('output_dir'):continue
            dest=Path(item['output_dir'])
            for row in item.get('result',{}).get('trials',[]):
                rp=dest/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
                if rp.exists():add_record(read(rp),row['trial_id']);assets.append(rp)
            for name in ['RESULTS.json','PREPARED.json','training_records.jsonl']:
                if(dest/name).exists():assets.append(dest/name)
        records.sort(key=lambda r:order[r['collection_context']['trial_id']])
        eligible={r['trial_id']for r in rows if r.get('label_eligible')is True and r.get('label')in (0,1)}
        if {r['collection_context']['trial_id']for r in records}!=eligible or len(records)!=len(eligible):raise ValueError('Training exports differ from actual eligible labels')
        actual={r['trial_id']for r in rows if r.get('physics_steps',0)>0}
        ready={r['trial_id']for r in queue['trials']}
        submitted=set()
        for marker in (output/'object_queues').glob('*.SUBMITTING.json'):
            submitted.update(read(marker)['trial_ids'])
        unknown=sorted((actual-eligible-set(certificates)) | ((submitted & ready)-actual))
        unexecuted=sorted(ready-actual-submitted)
        if unknown or unexecuted:failed=True
        counts={str(g):dict(target=100,labels=sum(r['gesture_id']==g and r.get('label_eligible')is True for r in rows),
            positive=sum(r['gesture_id']==g and r.get('label')==1 for r in rows),negative=sum(r['gesture_id']==g and r.get('label')==0 for r in rows))for g in GESTURES}
        for gesture,value in counts.items():
            value.update(shortfall=max(0,100-value['labels']),extra=max(0,value['labels']-100),depths={str(d):dict(target=25,labels=sum(r['gesture_id']==int(gesture) and r['extra_depth_index']==d and r.get('label_eligible')is True for r in rows))for d in DEPTHS})
            for depth in value['depths'].values():depth['shortfall']=max(0,25-depth['labels'])
        physical_steps=sum(r.get('physics_steps',0)for r in rows)
        result=dict(schema='anydex_boya_s40_collection_results_v1',status='infrastructure_stopped'if failed else'complete_with_null',complete=not failed,
            slot_count=640,objects=40,all_objects_training=True,actual_grasps=len(actual),physical_steps=physical_steps,
            inherited_actual_grasps=5,inherited_physics_steps=10500,new_actual_grasps=len(actual)-5,new_physical_steps=physical_steps-10500,
            eligible_labels=len(records),positives=sum(r.get('label')==1 for r in rows),negatives=sum(r.get('label')==0 for r in rows),
            preparation_nulls=queue['nulls'],terminal_certificates=certificates,terminal_nulls=len(certificates),
            unexecuted_ready_trial_ids=unexecuted,unknown_execution_trial_ids=unknown,physical_cost_complete=not unknown,counts=counts,trials=rows,
            new_CGR_forwards=38,inherited_CGR_forwards=2,new_training_updates=0,no_decision_head_loaded=True,
            wall_s=time.monotonic()-started,source_sha256_unchanged=True,input_sha256_unchanged=True)
        self.check()
        for fn in ['train_records.jsonl','all_records.jsonl']:
            with(output/fn).open('x')as f:
                for value in records:f.write(json.dumps(value,allow_nan=False)+'\n')
            assets.append(output/fn)
        save(output/'RESULTS.json',result);save(self.run/'collection_counts.json',{k:v for k,v in result.items()if k not in ['trials','terminal_certificates','preparation_nulls']})
        progress.update(status=result['status'],wall_s=result['wall_s']);save(output/'PROGRESS.json',progress)
        assets += [output/'RESULTS.json',output/'PROGRESS.json',self.run/'collection_counts.json']
        if failed:raise RuntimeError('Unknown execution/infrastructure evidence retained; no automatic replay')
        return dict(eligible_labels=len(records),physical_steps=physical_steps,new_physical_steps=physical_steps-10500,counts=counts),assets


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--scene-manifest',type=Path,default=ROOT/'outputs/s40_alignment_pilot_20261106_v1/scenes/objects.json')
    p.add_argument('--pilot',type=Path,default=ROOT/'outputs/s40_alignment_pilot_20261106_v1')
    p.add_argument('--calibration',type=Path,default=ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json')
    p.add_argument('--backend-config',type=Path,default=ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json')
    p.add_argument('--checkpoint',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18'))
    p.add_argument('--native-deps',type=Path,default=ROOT/'outputs/native_deps')
    p.add_argument('--device',default='cuda:0');p.add_argument('--geometry-workers',type=int,default=4)
    p.add_argument('--plan-workers',type=int,default=8);p.add_argument('--execution-workers',type=int,default=4)
    p.add_argument('--native-workers',type=int,default=4)
    p.add_argument('--phase',choices=['prepare','execute','all'],default='prepare')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    for key,maximum in [('geometry_workers',8),('plan_workers',16),('execution_workers',4),('native_workers',4)]:
        if not 1<=getattr(a,key)<=maximum:p.error('Invalid '+key)
    return a


def main():
    args=parse_args();run=args.run_dir.resolve();run.parent.mkdir(parents=True,exist_ok=True)
    with(run.parent/('.'+run.name+'.s40.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze(args);pipeline=Pipeline(run,protocol);pipeline.check()
        if args.dry_run:
            immutable(run/'DRY_PLAN.json',dict(protocol_sha256=sha(run/'protocol.json'),slots=640,inherited_slots=8,new_slots=632,
                inherited_actual_grasps=5,new_CGR_forwards_planned=38,valid_label_target_per_gesture=100,
                gesture_ids=GESTURES,extra_depth_indices=DEPTHS,no_test_objects=True,new_physics_steps=0,new_training_updates=0))
            pipeline.publish('frozen_dry_plan_only');print(json.dumps(read(run/'DRY_PLAN.json'),indent=2));return
        try:
            phases=[('native_assets',pipeline.native),('observations',pipeline.observations),('representation',pipeline.representation),
                    ('geometry',pipeline.geometry),('planning',pipeline.planning)]
            if args.phase=='execute':
                for name,_ in phases:
                    receipt=read(run/'stages'/(name+'.json'))
                    if receipt.get('complete')is not True or receipt['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('Execute requires completed frozen preparation')
                    verify(receipt['artifacts_sha256'])
            else:
                for name,fn in phases:pipeline.stage(name,fn)
            if args.phase in ['execute','all']:
                result=pipeline.stage('execution',pipeline.execution)
                pipeline.progress.update(physical_steps=result['physical_steps'],new_physical_steps=result['new_physical_steps'],eligible_labels=result['eligible_labels'])
                pipeline.publish('collection_complete')
            else:pipeline.publish('preparation_complete')
        except BaseException:
            pipeline.progress['error']=traceback.format_exc();pipeline.publish('infrastructure_stopped');raise


if __name__=='__main__':main()
