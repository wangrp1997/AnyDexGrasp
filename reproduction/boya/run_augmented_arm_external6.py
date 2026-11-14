"""Frozen new-object full-arm validation of the two existing Boya head recipes.

Explicit one-shot stages, no outcome-guided recovery, retraining or promotion.
Original/modified heads are compared on the SAME six scenes with augmentation.
Identity is an offline control only; this is not a physical augmentation ablation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import prepare_egad_external_scenes as assets
from planning_recovery import POLICY

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'reproduction/boya'
RUN=ROOT/'outputs/boya_augmented_arm_external6_20261114_v1'
AUDIT=ROOT/'outputs/boya_augmented_arm_audit_20261114_v1'
OLD_PLAN=ROOT/'outputs/boya_augmented_arm_planning_20261114_v2/RESULTS.json'
OLD=ROOT/'outputs/boya_fast6_20261111_v1'
IDS=tuple('egad_'+r+str(c) for r in 'ACE' for c in (2,5))
RECIPES=('original_mse','preactivation')
STAGES=('assets','initialize','observe','represent','score','plan','execute','media')
NEW_SOURCES=('prepare_augmented_arm.py','prepare_augmented_arm_assets.py',
    'prepare_augmented_arm_runtime.py','render_augmented_arm_observations.py',
    'compare_augmented_arm.py','run_augmented_arm_external6.py')
sha,save=assets.sha,assets.save
def read(p):return json.loads(Path(p).read_text())


def verify(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError('Frozen dependency changed: '+p)


def environment():
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
    overlay=ROOT/'outputs/native_deps/clarabel_0111_overlay'
    env['PYTHONPATH']=str(overlay)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    libs=[ROOT/'outputs/native_deps/os_libs',ROOT/'outputs/native_deps/openblas/lib',
        ROOT/'outputs/native_deps/cuda-13.0/lib64',Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages/torch/lib')]
    env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libs))+(os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def command(argv,log,timeout=3600):
    began=time.monotonic()
    with Path(log).open('xb') as out:
        proc=subprocess.run(list(map(str,argv)),cwd=ROOT,env=environment(),stdout=out,stderr=subprocess.STDOUT,timeout=timeout)
    receipt=dict(argv=list(map(str,argv)),returncode=proc.returncode,wall_s=time.monotonic()-began,log=str(log))
    save(Path(log).with_suffix('.process.json'),receipt)
    if proc.returncode:raise RuntimeError('Stage failed; preserve original output: '+str(log))
    return receipt


def freeze():
    if AUDIT.exists() or RUN.exists():raise FileExistsError('Fresh study required')
    audit_dir=AUDIT; audit_dir.mkdir()
    inventory_path=Path('/mnt/public/datasets/AnyDexGrasp/object_alignment_20261106_v1/egad_eval_inventory.json')
    inventory=read(inventory_path);assets.validate_inventory(inventory)
    exclusions=assets.exclusion_receipt([*assets.EXCLUSION_REFS,
        ROOT/'outputs/egad_external12_20261107_v1/objects.json',OLD/'objects.json'])
    byid={r['object_id']:r for r in inventory['objects']};selected=[]
    for oid in IDS:
        row=byid[oid[5:]];_,mesh=assets.scaled_visual(row['path'])
        digest=assets.meter_geometry_sha(mesh)
        if row['sha256'] in exclusions['source_hashes'] or digest in exclusions['meter_geometry_hashes']:
            raise ValueError('Object overlaps checked prior Boya source or metre geometry: '+oid)
        selected.append(dict(object_id=oid,source_object_id=oid[5:],source_mesh=row['path'],
            source_mesh_sha256=row['sha256'],meter_geometry_sha256=digest,
            source_identity_overlap=[],meter_geometry_overlap=[]))
    selection=dict(schema='boya_augmented_arm6_selection_audit_v1',objects=selected,
        rule='EGAD rows A,C,E x columns2,5 in grid order, no outcome/score/shape selection',
        exclusion_manifest_refs=exclusions['manifest_refs'],all_identity_and_meter_geometry_disjoint=True,
        source_inventory_ref=dict(path=str(inventory_path),sha256=sha(inventory_path)),
        boundary='Disjoint checked Boya source/digests; upstream CGR pretraining novelty and semantic shape equivalence unverified',
        physics_steps=0,network_forwards=0,training_updates=0)
    save(AUDIT/'SELECTION_AUDIT.json',selection)
    parent=ROOT/'outputs/boya_augmentation_physics_20261114_v1/STUDY_PROTOCOL.json'
    files=dict(read(parent)['sources_and_inputs']);files[str(parent)]=sha(parent)
    for name in NEW_SOURCES:
        path=B/name;files[str(path)]=sha(path)
    for p in (inventory_path,AUDIT/'SELECTION_AUDIT.json',*map(lambda r:Path(r['source_mesh']),selected)):
        files[str(p)]=sha(p)
    # Bind the isolated native audit dependency, rather than changing the shared environment.
    for p in sorted((ROOT/'outputs/native_deps/clarabel_0111_overlay').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:files[str(p)]=sha(p)
    verify(files)
    study=dict(schema='boya_augmented_arm6_predeclared_v1',object_order=list(IDS),recipe_order=list(RECIPES),
        primary_metric='original_mse successes / all6 new objects',secondary_metric='preactivation successes / same6 objects',
        between_recipe_comparison_paired_by_object=True,representation_branches=11,
        identity_forward_control_per_object=1,new_CGR_forwards_maximum=72,
        physical_conditions=['augmented11__'+r for r in RECIPES],identity_physical_control=False,
        augmentation_improvement_causal_claim_authorized=False,
        arm_budget_per_object_recipe=4,physical_attempts_per_object_recipe=1,max_actual_grasps=12,
        old_planning_gate='Complete zero-step A3/A6/E0 x two-recipe planning with >=1 ready case; otherwise stop before new object initialization',
        candidate_selection='First arm-ready from first4 full-hand-geometry admissions in frozen learned order',
        closure_mapping_threshold_models_training_and_physics_unchanged=True,
        no_plan_in_denominator=True,outcome_dependent_retry=False,training_writeback=False,default_promotion=False,
        initial_pose_policy='Original uniform native-axis AABB placement, measured settling and cold recreation',
        source_unit_to_m=.001,mass_kg=.1,friction=.6,planning_policy=POLICY,
        stop='Complete frozen queue; infrastructure errors halt subsequent waves, no silent retry/replacement/tuning',
        boundary='Independent Boya full-arm simulation using coarse convex proxies; not author real hardware paper success reproduction',
        frozen_files=files,created_unix_s=time.time())
    save(AUDIT/'STUDY_PROTOCOL.json',study)
    return study


def preparation():
    old=read(OLD_PLAN)
    if old.get('status')!='complete' or old.get('physical_steps')!=0 or old.get('errors'):
        raise ValueError('Old-object pure planning must finish without infrastructure error')
    if not any(r['selected'] for r in old['results']):
        raise ValueError('Planning gate failed: no full-arm ready old case; do not launch new objects')
    from prepare_augmented_arm_assets import prepare
    prepare(RUN,workers=3)
    protocol=copy.deepcopy(read(OLD/'protocol.json'))
    study=read(AUDIT/'STUDY_PROTOCOL.json');sel=read(AUDIT/'SELECTION_AUDIT.json')
    protocol.update(schema='anydex_boya_augmented_arm6_protocol_v1',objects=sel['objects'],object_count=6,
        selection_rule=sel['rule'],head_recipe='paired_frozen_original_and_preactivation',
        original_training_recipe='original_mse is primary; preactivation is explicit modified control',
        training_results={r:str(ROOT/('outputs/s40_code_recipe_20261107_v1' if r=='original_mse' else 'outputs/s40_preactivation_control_20261107_v1')/'TRAINING_RESULTS.json') for r in RECIPES},
        source_sha256={str(B/name):sha(B/name) for name in NEW_SOURCES},input_sha256=study['frozen_files'],
        study_ref=dict(path=str(AUDIT/'STUDY_PROTOCOL.json'),sha256=sha(AUDIT/'STUDY_PROTOCOL.json')),
        planning_gate_ref=dict(path=str(OLD_PLAN),sha256=sha(OLD_PLAN)),
        physical_attempts_per_object=1,physical_attempts_scope='Per frozen scoring recipe; at most2 total per object',
        sole_deployment_change='Original eleven-branch representation; two previously frozen heads evaluated separately',
        validation_boundary=study['boundary'],development_evidence=dict(old_augmented_arm_planning=str(OLD_PLAN)))
    protocol['representation']['branches']=11
    protocol['active_asset_policy'].update(source_scene_manifest=str(RUN/'scenes/objects.json'),
        selection_ref=dict(path=str(RUN/'scenes/SELECTION.json'),sha256=sha(RUN/'scenes/SELECTION.json')))
    from prepare_augmented_arm_runtime import validate_fast6_manifest
    validate_fast6_manifest(read(RUN/'scenes/objects.json'),protocol)
    save(RUN/'protocol.json',protocol)
    save(RUN/'EVALUATION_FREEZE.json',dict(protocol_sha256=sha(RUN/'protocol.json'),study_sha256=sha(AUDIT/'STUDY_PROTOCOL.json')))


def represent():
    objects=read(RUN/'objects.json');rows=[]
    for row in objects['objects']:
        oid=row['object_id'];obs=RUN/'observations'/oid/'observation.json'
        rows.append(dict(object_id=oid,observation_json=str(obs),observation_sha256=sha(obs),
            representation_dir=str(RUN/'identity'/oid)))
    save(RUN/'REPRESENTATION_MANIFEST.json',dict(objects=rows))
    command([sys.executable,'-B',B/'run_collection_representation.py','--manifest',RUN/'REPRESENTATION_MANIFEST.json',
        '--run-dir',RUN/'identity_batch','--device','cuda:0','--seed','20261108'],RUN/'identity_batch.log')
    parent=read(ROOT/'outputs/boya_augmentation_physics_20261114_v1/egad_A3/REPRESENTATION_PROTOCOL.json')
    for row in rows:
        oid=row['object_id'];path=RUN/oid;path.mkdir()
        protocol=copy.deepcopy(parent);protocol['frozen_files']=dict(read(AUDIT/'STUDY_PROTOCOL.json')['frozen_files'])
        protocol.update(observation_json=row['observation_json'],known_object='New-to-Boya external object '+oid,
            representation_dir=str(path/'representation'),identity_reference_dir=row['representation_dir'],
            parent_study=dict(path=str(AUDIT/'STUDY_PROTOCOL.json'),sha256=sha(AUDIT/'STUDY_PROTOCOL.json')))
        obs=read(row['observation_json']);protocol['frozen_files'][row['observation_json']]=sha(row['observation_json'])
        for v in obs['files'].values():protocol['frozen_files'][v['path']]=v['sha256']
        for p in Path(row['representation_dir']).glob('*.npz'):protocol['frozen_files'][str(p)]=sha(p)
        save(path/'REPRESENTATION_PROTOCOL.json',protocol)
        command([sys.executable,'-B',B/'run_augmented_representation.py','--protocol',path/'REPRESENTATION_PROTOCOL.json'],path/'representation.log')
    # The immutable representation helper's inherited 'known-scene' scope text
    # is conservative; study/selection refs establish this cohort's actual role.


def score():
    base=read(ROOT/'outputs/boya_augmentation_physics_20261114_v1/egad_A3/COMPARISON_PROTOCOL.json')
    jobs=[]
    for oid in IDS:
        path=RUN/oid;protocol=copy.deepcopy(base)
        protocol.update(schema='boya_augmented_arm6_offline_v1',object_id=oid,
            scope='Fixed new-to-Boya external object, no labels or physical outcomes',
            observation_json=str(RUN/'observations'/oid/'observation.json'),
            representation_dirs=dict(identity=str(RUN/'identity'/oid),augmented11=str(path/'representation')),
            output_dir=str(path/'comparison'),frozen_files=dict(read(AUDIT/'STUDY_PROTOCOL.json')['frozen_files']),
            parent_protocol=dict(path=str(RUN/'protocol.json'),sha256=sha(RUN/'protocol.json')))
        protocol.pop('preparation_checks',None)
        obs=read(protocol['observation_json'])
        for p in [Path(protocol['observation_json']),path/'representation/RESULTS.json',RUN/'identity'/oid/'RESULTS.json']:
            protocol['frozen_files'][str(p)]=sha(p)
        for v in obs['files'].values():protocol['frozen_files'][v['path']]=v['sha256']
        save(path/'COMPARISON_PROTOCOL.json',protocol);jobs.append(path)
    def one(path):
        return command([sys.executable,'-B',B/'compare_augmented_arm.py','--protocol',path/'COMPARISON_PROTOCOL.json'],path/'comparison.log')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for r in pool.map(one,jobs):print('SCORED',r['log'],flush=True)


def execute():
    from execute_collection_queue_v3 import install_temporal_contract
    from terminal_collection_outcome import certify_known_task_stop
    install_temporal_contract()
    planning=read(RUN/'planning/RESULTS.json')
    if planning['status']!='complete' or planning['errors']:raise ValueError('Complete planning required')
    if {(v['object_id'],v['recipe']) for v in planning['results']}!={(o,r) for o in IDS for r in RECIPES}:
        raise ValueError('Exactly twelve conditions required, including every null')
    conditions=[]
    for v in planning['results']:
        attempts=v['attempts']
        if v['arm_preparations']!=len(attempts) or len(attempts)>4:
            raise ValueError('Four-candidate budget violated')
        ready=[a for a in attempts if a['status']=='ready']
        if v['selected'] is not None and (len(ready)!=1 or attempts[-1]!=ready[0] or ready[0]['result']!=v['selected']):
            raise ValueError('Selection differs from first ready')
        if v['selected'] is None and ready:raise ValueError('Dropped ready condition')
        conditions.append(dict(object_id=v['object_id'],recipe=v['recipe'],selected=v['selected'],status=v['status']))
    out=RUN/'execution';out.mkdir();(out/'manifests').mkdir();(out/'runs').mkdir()
    frozen=dict(schema='boya_augmented_arm6_physical_choices_v1',protocol_sha256=sha(RUN/'protocol.json'),
        planning_sha256=sha(RUN/'planning/RESULTS.json'),conditions=conditions,
        physical_attempt_limit=12,outcome_dependent_retry=False,training_writeback=False)
    save(out/'ALL_CHOICES_FREEZE.json',frozen)
    # Check the isolated prerequisite before any Kit process or physical command.
    command([sys.executable,'-B','-c','import clarabel; assert clarabel.__version__ == "0.11.1"'],out/'dependency_check.log')
    selected=[v for v in conditions if v['selected']];got=[];errors=[];began=time.monotonic()
    def one(case):
        row=case['selected'];key=case['object_id']+'__'+case['recipe'];manifest=out/'manifests'/(key+'.json')
        save(manifest,dict(schema='anydex_boya_collection_queue_v1',complete=True,errors=[],inputs_unchanged=True,
            trials=[row],nulls=[],benchmark_only=True,training_writeback=False,
            parent_physical_choices_ref=dict(path=str(out/'ALL_CHOICES_FREEZE.json'),sha256=sha(out/'ALL_CHOICES_FREEZE.json'))))
        target=out/'runs'/key
        rec=command([sys.executable,'-B',B/'execute_collection_queue_v3.py','--manifest',manifest,'--output-dir',target,
            '--device','cpu','--mode','serial','--batch-size','1','--benchmark-only',
            '--paired-benchmark-id',sha(out/'ALL_CHOICES_FREEZE.json')],out/(key+'.log'))
        result=read(target/'RESULTS.json')
        if result['status'] not in ('complete','complete_with_null') or len(result['trials'])!=1:
            raise ValueError('Incomplete full-arm execution '+key)
        trial=result['trials'][0]
        if trial['trial_binding_sha256']!=row['trial_binding_sha256'] or trial['trial_id']!=row['trial_id']:
            raise ValueError('Physical binding mismatch')
        certificate=None if trial.get('label_eligible') else certify_known_task_stop(trial)
        return dict(object_id=case['object_id'],recipe=case['recipe'],trial=trial,
            terminal_certificate=certificate,actual_grasps=result['actual_grasps'],
            physical_steps=result['physical_steps'],process=rec)
    # Do not submit a later wave after an infrastructure error in this wave.
    for start in range(0,len(selected),3):
        with ThreadPoolExecutor(max_workers=3) as pool:
            active={pool.submit(one,v):v for v in selected[start:start+3]}
            for future in as_completed(active):
                try:got.append(future.result())
                except Exception:errors.append(dict(case=active[future],error=traceback.format_exc()))
                save(out/'PROGRESS.json',dict(status='executing',finished=len(got),errors=errors,planned=len(selected)))
        if errors:break
    metrics={}
    for recipe in RECIPES:
        rows=[v for v in got if v['recipe']==recipe]
        metrics[recipe]=dict(objects=6,actual_attempts=sum(v['actual_grasps'] for v in rows),
            successes=sum(v['trial'].get('label')==1 for v in rows),
            actual_failures=sum(v['trial'].get('label')==0 or v['terminal_certificate'] is not None for v in rows),
            no_plan=sum(v['recipe']==recipe and v['selected'] is None for v in conditions))
    report=dict(schema='boya_augmented_arm6_execution_results_v1',status='complete' if not errors else 'infrastructure_error',
        conditions=conditions,results=got,metrics=metrics,errors=errors,
        actual_attempts=sum(v['actual_grasps'] for v in got),physical_steps=sum(v['physical_steps'] for v in got),
        physical_choices_sha256=sha(out/'ALL_CHOICES_FREEZE.json'),
        benchmark_only=True,training_writeback=False,outcome_dependent_retry=False,
        wall_s=time.monotonic()-began,final_success_rates_available=not errors)
    save(out/'RESULTS.json',report)
    if errors:raise RuntimeError('Physical stage incomplete; no automatic retry')


def media():
    out=RUN/'videos';out.mkdir();report=read(RUN/'execution/RESULTS.json')
    if report['status']!='complete':raise ValueError('Complete physical stage required')
    def one(v):
        key=v['object_id']+'__'+v['recipe'];row=v['trial'];target=out/key
        rec=command([sys.executable,'-B',B/'render_native.py','--execution',row['execution_dir'],
            '--plan-dir',row['source_plan_dir'],'--output-dir',target,'--object-label',key+' label='+str(row['label']),
            '--render-gpu','0','--include-failed-trace'],out/(key+'.log'))
        video=target/'isaac_full_arm_pickup.mp4'
        if not video.is_file():raise ValueError('Missing video '+key)
        return dict(object_id=v['object_id'],recipe=v['recipe'],label=row['label'],path=str(video),sha256=sha(video),process=rec)
    with ThreadPoolExecutor(max_workers=2) as pool:videos=list(pool.map(one,report['results']))
    page=['<!doctype html><meta charset="utf-8"><title>Boya augmented full-arm six objects</title>',
        '<h1>Six new Boya objects, full arm, frozen paired heads</h1>',
        '<p>Every object remains in each denominator, including no plan. Approximate convex physics, no retries.</p>',
        '<pre>'+json.dumps(report['metrics'],indent=2)+'</pre>']
    for v in videos:
        page.append('<h2>'+v['object_id']+' '+v['recipe']+' label='+str(v['label'])+'</h2><video controls width="960" src="videos/'+Path(v['path']).relative_to(out).as_posix()+'"></video>')
    (RUN/'index.html').write_text('\n'.join(page))
    save(out/'RESULTS.json',dict(status='complete',videos=videos,replay_physics_steps=0))


def stage(name):
    study=read(AUDIT/'STUDY_PROTOCOL.json');verify(study['frozen_files'])
    progress_path=AUDIT/'PROGRESS.json'
    progress=read(progress_path) if progress_path.exists() else dict(status='prepared',stages=[])
    if any(v['stage']==name for v in progress['stages']):raise ValueError('Stage already attempted; no silent retry: '+name)
    completed=[v['stage'] for v in progress['stages'] if v.get('returncode')==0]
    if completed!=list(STAGES[:STAGES.index(name)]):raise ValueError('Earlier stages must complete in order')
    if name!='assets':
        if sha(RUN/'protocol.json')!=read(RUN/'EVALUATION_FREEZE.json')['protocol_sha256']:raise ValueError('Protocol changed')
    receipt=dict(stage=name,started_unix_s=time.time(),returncode=None);progress['stages'].append(receipt)
    progress['status']='running';save(progress_path,progress);began=time.monotonic()
    try:
        if name=='assets':preparation()
        elif name=='initialize':command([sys.executable,'-B',B/'prepare_augmented_arm_runtime.py',
            '--scene-manifest',RUN/'scenes/objects.json','--run-dir',RUN,'--workers','3'],RUN/'initialize.log')
        elif name=='observe':command([sys.executable,'-B',B/'render_augmented_arm_observations.py',
            '--objects-json',RUN/'objects.json','--output-dir',RUN/'observations'],RUN/'observe.log')
        elif name=='represent':represent()
        elif name=='score':score()
        elif name=='plan':command([sys.executable,'-B',B/'prepare_augmented_arm.py','--comparison',
            *[RUN/o/'comparison' for o in IDS],'--native-preflight',RUN/'native_preprocessing/SUMMARY.json',
            '--output',RUN/'planning','--workers','3'],RUN/'planning.log')
        elif name=='execute':execute()
        elif name=='media':media()
        verify(study['frozen_files']);receipt['returncode']=0;progress['status']='complete' if name=='media' else 'stage_complete'
    except BaseException:
        receipt.update(returncode=1,error=traceback.format_exc());progress['status']='failed';raise
    finally:
        receipt['wall_s']=time.monotonic()-began;save(progress_path,progress)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('freeze',*STAGES),required=True)
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:stage(args.stage)
