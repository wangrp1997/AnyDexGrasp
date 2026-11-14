"""Frozen finite multi-scene lift collection, with explicit once-only stages.

Freeze writes no scene or simulator. Run launches declared external processes;
prepare concurrency is at most three, while capture/CGR/execution are serial on
GPU0. No training, old scoring head or candidate fallback is dispatched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
TASK = 'declared_boya_clutter_lift_v1'
SCHEMA = 'anydex_boya_author_collection_batch_v1'
QUEUE_SCHEMA = 'anydex_boya_author_collection_scene_queue_v1'
GESTURES = (1, 6, 7, 8)
SOURCE_TYPES = {1: 1, 6: 2, 7: 3, 8: 4}
DEFAULT_ENV = dict(CUDA_VISIBLE_DEVICES='0', NVIDIA_VISIBLE_DEVICES='0',
    PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OMP_NUM_THREADS='4', OMNI_KIT_ACCEPT_EULA='YES')


class InterruptedChildUnresolved(RuntimeError):
    pass


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def verify(bindings):
    for entry in bindings.values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen batch dependency changed: '+entry['path'])


def normalized_bindings(values):
    return {name: (dict(path=str(Path(name).resolve()), sha256=value)
                   if isinstance(value, str) else value) for name, value in values.items()}


def _safe_id(value):
    return isinstance(value, str) and bool(value) and Path(value).name == value and value not in ('.', '..')


def validate_queue(path):
    queue = read(path)
    if queue.get('schema') != QUEUE_SCHEMA or not queue.get('scenes'):
        raise ValueError('Explicit nonempty frozen scene queue required')
    verify(normalized_bindings(queue.get('inputs', {})))
    verify(normalized_bindings(queue.get('sources', {})))
    identities, split_ids, split_meshes = set(), {'fit': set(), 'internal': set()}, {'fit': set(), 'internal': set()}
    for row in queue['scenes']:
        if not _safe_id(row['scene_id']) or row['scene_id'] in identities or row['split'] not in split_ids:
            raise ValueError('Ordered unique safe scene IDs and explicit fit/internal split required')
        identities.add(row['scene_id'])
        verify({'spec': row['scene_spec']})
        spec=read(row['scene_spec']['path'])
        if row.get('status',spec.get('status','ready')) not in ('ready','placement_null'):
            raise ValueError('Scene queue must declare ready or placement_null')
        if spec.get('status',row.get('status','ready')) != row.get('status',spec.get('status','ready')):
            raise ValueError('Scene queue placement status differs from frozen spec')
        ids, digests = row['source_object_ids'], row['source_object_digests']
        if (not isinstance(ids, list) or len(ids) != 3 or len(set(ids)) != 3
                or any(not isinstance(i, str) or not _safe_id(i) for i in ids)
                or set(digests) != set(ids) or any(not isinstance(h, str) or len(h) != 64 for h in digests.values())):
            raise ValueError('Exactly three fully bound source objects per scene required')
        split_ids[row['split']].update(ids)
        split_meshes[row['split']].update(digests.values())
    if split_ids['fit'] & split_ids['internal'] or split_meshes['fit'] & split_meshes['internal']:
        raise ValueError('All scene objects, including nonprimary bodies, must be split-disjoint')
    return queue


def freeze(queue_path, output, *, calibration, cloud_manifest, backend,
           fit_per_gesture=1000, internal_per_gesture=200, prepare_workers=3,
           scene_entry=None, capture_entry=None, execution_entry=None, audit_entry=None, checkpoint=None, native_deps=None,
           domain=None, stage_timeout_s=3600., python=None):
    """Freeze the finite budget; never build/capture/plan/execute/train."""
    if (type(fit_per_gesture) is not int or fit_per_gesture < 1
            or type(internal_per_gesture) is not int or internal_per_gesture < 1
            or type(prepare_workers) is not int or not 1 <= prepare_workers <= 3
            or not np.isfinite(stage_timeout_s) or stage_timeout_s <= 0):
        raise ValueError('Positive label quotas/timeout and prepare_workers1..3 required')
    queue_path, output = Path(queue_path).resolve(), Path(output).resolve()
    queue = validate_queue(queue_path)
    scene_entry = Path(scene_entry or HERE/'author_collection_scenes.py').resolve()
    capture_entry = Path(capture_entry or HERE/'author_collection_capture.py').resolve()
    execution_entry = Path(execution_entry or HERE/'author_collection_execute.py').resolve()
    audit_entry = Path(audit_entry or HERE/'audit_author_collection_runtime.py').resolve()
    sources = ['author_collection_batch.py', 'author_collection_export.py', 'author_lift_pipeline.py',
        'author_collect.py', 'author_lift.py', 'author_lift_plan.py', 'author_pipeline.py',
        'author_transport.py', 'author_collection_bridge.py', 'author_action_semantics.py',
        'author_protocol.py', 'author_collision.py', 'author_hand_clouds.py',
        'author_joint_home_driver.py', 'author_joint_home_route.py', 'author_home_retime.py',
        'author_home_route.py', 'native_arm.py', 'run_collection_representation.py',
        'run_pilot_representation.py', 'asset_adapter.py', 'trial_record.py', 'boya_prehead_domain.py']
    source_bindings = {n: binding(HERE/n) for n in sources}
    source_bindings.update(scene_entry=binding(scene_entry), capture_entry=binding(capture_entry), execution_entry=binding(execution_entry), audit_entry=binding(audit_entry))
    inputs = {k: binding(p) for k, p in dict(queue=queue_path, calibration=calibration,
        cloud_manifest=cloud_manifest, backend=backend,
        checkpoint=checkpoint or '/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18',
        python=python or sys.executable).items()}
    # Keep the venv invocation path: resolving its Python symlink would bypass
    # the virtualenv even though the underlying executable bytes are identical.
    inputs['python']['path']=str(Path(python or sys.executable).absolute())
    for name, entry in normalized_bindings(queue.get('inputs', {})).items():
        inputs['queue:'+name] = entry
    for name, entry in normalized_bindings(queue.get('sources', {})).items():
        source_bindings['queue:'+name] = entry
    for index, row in enumerate(queue['scenes']):
        inputs['scene_spec:'+str(index)] = row['scene_spec']
    if queue.get('split_manifest'):inputs['split_manifest']=queue['split_manifest']
    from author_pipeline import DOMAIN
    protocol = dict(schema=SCHEMA, task=TASK, inputs=inputs, sources=source_bindings,
        queue=queue, quotas=dict(fit=fit_per_gesture, internal=internal_per_gesture),
        gestures=list(GESTURES), source_types=SOURCE_TYPES, prepare_workers=prepare_workers,
        domain=domain if domain is not None else DOMAIN,
        stage_timeout_s=float(stage_timeout_s), native_deps=str(Path(native_deps or ROOT/'outputs/native_deps').resolve()),
        env=DEFAULT_ENV, inherited_env={k: os.environ.get(k, '') for k in ('PYTHONPATH','LD_LIBRARY_PATH')},
        total_scene_budget=len(queue['scenes']), total_gesture_attempt_budget=4*len(queue['scenes']),
        max_candidates_per_gesture=1, max_executions_per_gesture=1, retry_policy='No started stage is relaunched',
        argv_shapes=dict(scene='build --scene-spec PATH --output-dir DIR',
            capture=('capture 'if capture_entry.name=='author_scene.py'else'')+'--scene PATH --output-dir DIR',
            execute=('execute 'if execution_entry.name=='author_lift_pipeline.py'else'')+'--prepared-dir DIR',
            audit='--prepared-dir DIR --split-manifest PATH --output PATH'),
        stopping_policy='Each gesture/split retains the first quota eligible labels in immutable scene order; no success or class balancing',
        old435_merged=False, old_head_forwards=0, automatic_training=False,
        physics_concurrency=1, gpu=0, renderer_policy='Existing CPU-physics Kit entries use their default renderer GPU0; CUDA/NVIDIA visibility is frozen to0',
        started_unix_s=time.time())
    verify(inputs); verify(source_bindings)
    output.mkdir(parents=True, exist_ok=False)
    save(output/'BATCH_PROTOCOL.json', protocol, exclusive=True)
    save(output/'PROGRESS.json', _progress(protocol, []))
    return protocol


@contextmanager
def _lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another batch driver owns '+str(path)) from error
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def _environment(protocol):
    env = dict(os.environ, **protocol['inherited_env'], **protocol['env'])
    env['PYTHONPATH'] = str(HERE)+os.pathsep+str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env['PYTHONPATH'] else '')
    env['CANONDEX_COLLISION_PARTITION_CACHE'] = str(ROOT/'outputs/boya_collision_partition_cache')
    return env


def _stage(directory, command, expected, protocol):
    """One external attempt. A running receipt on resume becomes infrastructure null."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'STAGE.json'
    if path.exists():
        result = read(path)
        if result['command'] != list(map(str, command)):
            raise ValueError('Stage command differs from its frozen first attempt')
        if result['status'] == 'running':
            pid=result.get('child_pid')
            alive=bool(pid and Path('/proc',str(pid)).exists())
            result.update(status='infrastructure_failure', error='interrupted_stage_not_relaunched', complete=False,
                resume_blocked=alive or pid is None,
                interruption_scope='An alive or unknown orphan must be reviewed; no new stage may start')
            save(path, result)
        if result.get('resume_blocked'):raise InterruptedChildUnresolved('Interrupted child lifetime requires review: '+str(path))
        if result['status'] == 'complete': verify(result['outputs'])
        return result
    verify(protocol['inputs']); verify(protocol['sources'])
    result = dict(status='running', complete=False, command=list(map(str,command)),
        started_unix_s=time.time(), outputs={}, retry_count=0)
    save(path, result, exclusive=True); begin=time.monotonic()
    fatal=None
    try:
        with (directory/'process.log').open('xb') as log:
            process = subprocess.Popen(result['command'], cwd=ROOT, env=_environment(protocol),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['child_pid']=process.pid;save(path,result)
            try:
                process.wait(timeout=protocol['stage_timeout_s'])
            except BaseException:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait(timeout=5.)
                raise
        result['returncode'] = process.returncode
        if process.returncode != 0:
            raise RuntimeError('Declared subprocess returned '+str(process.returncode))
        result['outputs'] = {str(p): binding(p) for p in expected}
        verify(protocol['inputs']); verify(protocol['sources'])
        result.update(status='complete', complete=True)
    except BaseException as error:
        result.update(status='infrastructure_failure', complete=False, error=repr(error), traceback=traceback.format_exc())
        if isinstance(error,(KeyboardInterrupt,SystemExit)):fatal=error
    result['wall_s'] = time.monotonic()-begin
    result['log'] = binding(directory/'process.log') if (directory/'process.log').exists() else None
    save(path,result)
    if fatal is not None:raise fatal
    return result


def _scene_objects(path, row):
    scene=read(path)
    if scene.get('schema') != 'anydex_boya_author_lift_scene_v1' or scene.get('task') != TASK or scene.get('bin','missing') is not None:
        raise ValueError('Only explicit bin-free lift scenes can be collected')
    actual={r['object_id']: r.get('source_visual_mesh_sha256',r['source_mesh_sha256']) for r in scene['objects']}
    if actual != row['source_object_digests'] or set(actual) != set(row['source_object_ids']):
        raise ValueError('Scene objects differ from frozen whole-scene split inputs')
    if scene['scene_id'] != row['scene_id']:
        raise ValueError('Scene identity changed')
    if scene.get('dataset_split',row['split'])!=row['split']:
        raise ValueError('Scene dataset split differs from its complete source group')
    for obj in scene['objects']:
        if sha(obj['source_mesh']) != obj['source_mesh_sha256']:
            raise ValueError('Scene source visual mesh changed')
        if obj.get('source_visual_mesh') and sha(obj['source_visual_mesh'])!=obj['source_visual_mesh_sha256']:
            raise ValueError('Original dataset source visual mesh changed')
    return scene


def validate_trial(prepared_dir, row):
    """Recheck actual eligible record, full-scene identity and every SHA binding."""
    import author_lift_pipeline as pipeline
    import author_collect as collector
    prepared_dir=Path(prepared_dir)
    protocol=read(prepared_dir/'LIFT_PIPELINE_PROTOCOL.json')
    ready=read(prepared_dir/'READY.json')
    preparation=read(prepared_dir/'LIFT_PIPELINE_RESULTS.json')
    if preparation.get('status') != 'ready' or preparation.get('ready_sha256') != sha(prepared_dir/'READY.json'):
        raise ValueError('Actual lift-task ready preparation required')
    pipeline._verify_protocol(protocol)
    verify({k:ready[k] for k in ('protocol','scene','candidate','plan','commands','command_binding')})
    scene=_scene_objects(ready['scene']['path'],row)
    plan=read(ready['plan']['path']);pipeline.common._verify_plan_dependencies(plan)
    pipeline._verify_plan_artifacts(Path(ready['plan']['path']).parent,plan)
    candidate=read(ready['candidate']['path']);pipeline._candidate(candidate)
    commands=read(ready['command_binding']['path'])
    if commands['execution_spec']['original_object_ids'] != [r['object_id']for r in scene['objects']]:
        raise ValueError('Actual command must bind every original scene body')
    record_path=prepared_dir/'COLLECTION_RECORD.json';record=read(record_path)
    verify({'evaluation':record['evaluation_binding']})
    recomputed=collector.collect_execution(candidate,commands,record['evaluation_binding']['path'])
    if record != recomputed or record.get('formal_training_eligible') is not True or type(record.get('label')) is not int or record['label'] not in (0,1):
        raise ValueError('Eligible executed binary label must survive independent binding recomputation')
    evaluation=read(record['evaluation_binding']['path'])
    if evaluation.get('original_object_ids') != [r['object_id']for r in scene['objects']]:
        raise ValueError('Evaluation body identities differ from scene')
    for key,name in (('trace_sha256','MULTI_OBJECT_TRACE.json'),('contacts_sha256','MULTI_OBJECT_CONTACTS.json')):
        artifact=Path(record['evaluation_binding']['path']).parent/name
        if sha(artifact) != evaluation[key]:
            raise ValueError('Actual all-body evaluation trace changed')
        pointer=read(artifact)
        if pointer.get('schema')=='anydex_boya_contact_journal_pointer_v1':
            verify({'contact_data':pointer['data']})
            if Path(pointer['data']['path']).stat().st_size!=pointer['data']['bytes']:
                raise ValueError('Lossless contact journal file size changed')
    features=np.asarray(candidate['canonical_cgr480'],dtype=np.float32)
    identity=candidate['canonical_cgr480_identity']
    if (features.shape != (480,) or not np.isfinite(features).all()
            or hashlib.sha256(features.tobytes()).hexdigest() != identity['c_order_bytes_sha256']
            or candidate['canonicalization'].get('applied_once') is not True):
        raise ValueError('Actual once-canonicalized float32 CGR480 must be bound')
    target=candidate['raw_depth_index']*4+candidate['extra_depth_index']
    if type(target) is not int or not 0<=target<20:
        raise ValueError('Original selected output raw_depth*4+extra required')
    ids=sorted(row['source_object_ids'])
    return dict(record=binding(record_path), candidate=binding(ready['candidate']['path']),
        evaluation=record['evaluation_binding'], commands=ready['commands'], command_binding=ready['command_binding'],
        scene=ready['scene'], label=record['label'], features=features.tolist(), target_index=target,
        scene_object_ids=ids,scene_mesh_sha256=[row['source_object_digests'][i]for i in ids],
        mesh_sha256=row['source_object_digests'][scene['target_object_id']],
        candidate_sha256=candidate['candidate_sha256'],gesture_id=candidate['gesture_id'],
        actual_physics_steps=evaluation['total_physics_steps'])


def certify_audit(audit,trial,row):
    """Require the independent all-body runtime audit before admitting a label."""
    ids=audit.get('original_object_ids',[])
    digests=audit.get('all_scene_source_mesh_sha256',[])
    if (not isinstance(ids,list) or len(ids)!=3 or len(set(ids))!=3
            or not isinstance(digests,list) or len(digests)!=3
            or dict(zip(ids,digests))!=row['source_object_digests']
            or audit.get('status')!='pass' or audit.get('formal_training_eligible')is not True
            or type(audit.get('label'))is not int or audit['label']!=trial['label']
            or audit.get('split')!=row['split']):
        raise ValueError('Independent all-body runtime/split audit did not certify this actual label')


def _progress(protocol, rows):
    counts={split:{str(g):0 for g in GESTURES} for split in ('fit','internal')}
    positive={split:{str(g):0 for g in GESTURES} for split in counts}
    for row in rows:
        if row.get('label') is not None:
            counts[row['split']][str(row['gesture_id'])]+=1
            positive[row['split']][str(row['gesture_id'])]+=row['label']
    captures={r['queue_index']:r['artifact_bindings']['capture_stage']for r in rows if 'capture_stage'in r.get('artifact_bindings',{})}
    completed_captures=sum(read(p['path']).get('status')=='complete'for p in captures.values())
    met=all(counts[s][str(g)]>=protocol['quotas'][s]for s in counts for g in GESTURES)
    return dict(schema=SCHEMA,task=TASK,status='quota_met'if met else 'collecting_or_budget_pending',
        eligible_labels=counts,positive_labels=positive,ledger_rows=len(rows),
        actual_physics_steps=sum(r.get('actual_physics_steps',0)for r in rows),
        actual_physics_steps_scope='Known final execution receipt only; excludes initialization; unknown starts are not zero-step claims',
        captured_initialization_ticks=completed_captures*750,
        captured_initialization_object_steps=completed_captures*750*3,
        initialization_cost_scope='Complete captures only; failed/interrupted capture costs are not included in this lower bound',
        execution_starts_with_unknown_steps=sum(bool(r.get('execution_steps_unknown'))for r in rows),
        training_updates=0,old_head_forwards=0,old435_merged=False,
        quota_met=met,total_scene_budget=protocol['total_scene_budget'],total_gesture_attempt_budget=protocol['total_gesture_attempt_budget'])


def ledger(batch_dir, protocol):
    rows=[];verified_stages=set()
    for p in sorted((Path(batch_dir)/'ledger').glob('*.json')):
        row=read(p)
        if row['protocol_sha256']!=sha(Path(batch_dir)/'BATCH_PROTOCOL.json'):
            raise ValueError('Ledger belongs to another frozen protocol')
        verify(row.get('artifact_bindings',{}))
        for entry in row.get('artifact_bindings',{}).values():
            if Path(entry['path']).name=='STAGE.json'and entry['path']not in verified_stages:
                verify(read(entry['path']).get('outputs',{}));verified_stages.add(entry['path'])
        rows.append(row)
    return rows


def _terminal(batch,protocol,index,row,gesture,status,*,trial=None,error=None,artifacts=None,steps=0,steps_unknown=False):
    verify(protocol['inputs']);verify(protocol['sources'])
    value=dict(schema=SCHEMA,task=TASK,protocol_sha256=sha(batch/'BATCH_PROTOCOL.json'),
        queue_index=index,scene_id=row['scene_id'],split=row['split'],gesture_id=gesture,
        trial_id=f'{row["scene_id"]}:g{gesture}',status=status,label=None,actual_physics_steps=steps,execution_steps_unknown=steps_unknown,
        source_object_ids=row['source_object_ids'],source_object_digests=row['source_object_digests'],
        scene_spec=row['scene_spec'],artifact_bindings=artifacts or {},error=error)
    if trial is not None:
        if trial['gesture_id']!=gesture:raise ValueError('Trial gesture differs from queued gesture')
        value.update(label=trial['label'],trial=trial,actual_physics_steps=trial['actual_physics_steps'])
        value['artifact_bindings'].update({k:trial[k]for k in ('record','candidate','evaluation','commands','command_binding','scene')})
    save(batch/'ledger'/f'{index:07d}_g{gesture}.json',value,exclusive=True)
    return value


def _finish_gesture(batch,protocol,index,row,g,directory,prepared_stage,common_artifacts):
    prepared=directory/f'g{g}';artifacts=dict(common_artifacts)
    artifacts['prepare_stage']=binding(directory/f'stages/prepare_g{g}/STAGE.json')
    execution={}
    try:
        if prepared_stage['status']!='complete':
            return _terminal(batch,protocol,index,row,g,'infrastructure_failure',error=prepared_stage.get('error'),artifacts=artifacts)
        prep=read(prepared/'LIFT_PIPELINE_RESULTS.json');artifacts['preparation']=binding(prepared/'LIFT_PIPELINE_RESULTS.json')
        if prep['status']!='ready':
            if prep['status']not in ('no_candidate','null_geometry','infrastructure_failure'):raise ValueError('Unknown lift preparation status')
            return _terminal(batch,protocol,index,row,g,prep['status'],artifacts=artifacts)
        command=[protocol['inputs']['python']['path'],'-B',protocol['sources']['execution_entry']['path']]
        if Path(command[-1]).name=='author_lift_pipeline.py':command.append('execute')
        stage=_stage(directory/f'stages/execute_g{g}',command+['--prepared-dir',str(prepared)],
            [prepared/'LIFT_PIPELINE_EXECUTION.json'],protocol)
        artifacts['execute_stage']=binding(directory/f'stages/execute_g{g}/STAGE.json')
        execution=read(prepared/'LIFT_PIPELINE_EXECUTION.json')if(prepared/'LIFT_PIPELINE_EXECUTION.json').exists()else{}
        if stage['status']!='complete':
            return _terminal(batch,protocol,index,row,g,'infrastructure_failure',error=stage.get('error'),artifacts=artifacts,steps=execution.get('physics_steps',0),
                steps_unknown=execution.get('status')in(None,'starting')or not(prepared/'execution/LIFT_EVALUATION.json').exists())
        if not (prepared/'COLLECTION_RECORD.json').exists() or read(prepared/'COLLECTION_RECORD.json').get('label')is None:
            return _terminal(batch,protocol,index,row,g,'executed_null',artifacts=artifacts,steps=execution.get('physics_steps',0))
        split=protocol['queue'].get('split_manifest')
        if not split:raise ValueError('Explicit frozen full-scene split manifest required for runtime audit')
        audit_path=prepared/'RUNTIME_AUDIT.json'
        audit_stage=_stage(directory/f'stages/audit_g{g}',[protocol['inputs']['python']['path'],'-B',
            protocol['sources']['audit_entry']['path'],'--prepared-dir',str(prepared),
            '--split-manifest',split['path'],'--output',str(audit_path)],[audit_path],protocol)
        artifacts['audit_stage']=binding(directory/f'stages/audit_g{g}/STAGE.json')
        if audit_path.exists():artifacts['runtime_audit']=binding(audit_path)
        if audit_stage['status']!='complete':
            raise ValueError('Independent runtime audit infrastructure failure: '+str(audit_stage.get('error')))
        verify(protocol['inputs']);verify(protocol['sources'])
        trial=validate_trial(prepared,row)
        certify_audit(read(audit_path),trial,row)
        return _terminal(batch,protocol,index,row,g,'labelled',trial=trial,artifacts=artifacts)
    except BaseException as error:
        path=prepared/'LIFT_PIPELINE_EXECUTION.json'
        if path.exists():execution=read(path)
        result=_terminal(batch,protocol,index,row,g,'infrastructure_failure',error=repr(error),artifacts=artifacts,steps=execution.get('physics_steps',0),
            steps_unknown='execute_stage'in artifacts and execution.get('status')in(None,'starting'))
        if isinstance(error,(KeyboardInterrupt,SystemExit,InterruptedChildUnresolved)):raise
        return result


def run(batch_dir, *, max_scenes=None):
    """Resume only unstarted stages in the frozen prefix; never retry a start."""
    batch=Path(batch_dir).resolve();protocol=read(batch/'BATCH_PROTOCOL.json')
    if protocol.get('schema')!=SCHEMA or protocol.get('task')!=TASK:raise ValueError('Explicit lift batch required')
    if max_scenes is not None and (type(max_scenes)is not int or max_scenes<1):raise ValueError('Positive max_scenes prefix required')
    with _lock(batch/'DRIVER.lock'), _lock(ROOT/'outputs/author_collection_gpu0.lock'):
        verify(protocol['inputs']);verify(protocol['sources'])
        for stage_path in (batch/'scenes').glob('*/stages/*/STAGE.json'):
            old=read(stage_path)
            if old.get('resume_blocked'):raise InterruptedChildUnresolved('Interrupted child lifetime requires review: '+str(stage_path))
        rows=ledger(batch,protocol);done={(r['queue_index'],r['gesture_id'])for r in rows}
        if len(done)!=len(rows):raise ValueError('Duplicate ledger trial')
        py=protocol['inputs']['python']['path'];scenes=protocol['queue']['scenes']
        for index,row in enumerate(scenes[:max_scenes]if max_scenes else scenes):
            counts=_progress(protocol,rows)['eligible_labels']
            active=[]
            for g in GESTURES:
                if (index,g)in done:continue
                if counts[row['split']][str(g)]>=protocol['quotas'][row['split']]:
                    rows.append(_terminal(batch,protocol,index,row,g,'quota_reached'))
                    done.add((index,g))
                else:active.append(g)
            if not active:continue
            spec=read(row['scene_spec']['path'])
            if spec.get('status',row.get('status','ready'))=='placement_null':
                for g in active:
                    rows.append(_terminal(batch,protocol,index,row,g,'placement_null',
                        error=spec.get('reason',spec.get('placement_failure')),artifacts={'scene_spec':row['scene_spec']}))
                save(batch/'PROGRESS.json',_progress(protocol,rows))
                save(batch/'LEDGER.json',dict(protocol_sha256=sha(batch/'BATCH_PROTOCOL.json'),rows=rows))
                continue
            directory=batch/'scenes'/f'{index:07d}_{row["scene_id"]}'
            build=directory/'build';capture=directory/'capture';rep=directory/'representation'
            common_artifacts={}
            try:
                stage=_stage(directory/'stages/build',[py,'-B',protocol['sources']['scene_entry']['path'],'build',
                    '--scene-spec',row['scene_spec']['path'],'--output-dir',str(build)],[build/'scene.json'],protocol)
                common_artifacts['build_stage']=binding(directory/'stages/build/STAGE.json')
                if stage['status']!='complete':raise RuntimeError('scene_build:'+str(stage.get('error')))
                _scene_objects(build/'scene.json',row)
                capture_args=[py,'-B',protocol['sources']['capture_entry']['path']]
                if Path(protocol['sources']['capture_entry']['path']).name=='author_scene.py':capture_args.append('capture')
                capture_outputs=[capture/'RESULTS.json',capture/'observation.json',capture/'observed_scene.json']
                if Path(protocol['sources']['capture_entry']['path']).name in ('author_scene.py','author_collection_capture.py'):
                    capture_outputs.extend(capture/n for n in ('initialization_trace.npz','native_rgbd.npz',
                        'CONTACT_SENSOR_CONFIG.json','CONTACT_SENSOR_CHECK.json','render_geometry_check.json',
                        'color.png','depth.png','meta.mat','PROGRESS.json'))
                    if Path(protocol['sources']['capture_entry']['path']).name=='author_collection_capture.py':
                        capture_outputs.extend(capture/n for n in ('CONTACT_JOURNAL.json','initialization_contacts.jsonl.gz','CAPTURE_OVERLAY.json'))
                    else:capture_outputs.append(capture/'initialization_contacts.json')
                stage=_stage(directory/'stages/capture',capture_args+['--scene',str(build/'scene.json'),'--output-dir',str(capture)],capture_outputs,protocol)
                common_artifacts['capture_stage']=binding(directory/'stages/capture/STAGE.json')
                if stage['status']!='complete' or read(capture/'RESULTS.json').get('status')!='complete':
                    raise RuntimeError('scene_capture:'+str(stage.get('error')))
                _scene_objects(capture/'observed_scene.json',row)
                obs=read(capture/'observation.json')
                if obs['scene_sha256']!=sha(capture/'observed_scene.json') or Path(obs['scene_json']).resolve()!=(capture/'observed_scene.json').resolve():
                    raise ValueError('Actual observation scene binding changed')
                manifest=directory/'REPRESENTATION_MANIFEST.json'
                if not manifest.exists():save(manifest,dict(objects=[dict(object_id=row['scene_id'],observation_json=str(capture/'observation.json'),observation_sha256=sha(capture/'observation.json'))]),exclusive=True)
                target=rep/'representations'/row['scene_id']
                stage=_stage(directory/'stages/representation',[py,'-B',str(HERE/'run_collection_representation.py'),
                    '--manifest',str(manifest),'--run-dir',str(rep),'--device','cuda:0','--seed',str(protocol['queue']['seed']),
                    '--checkpoint',protocol['inputs']['checkpoint']['path'],'--native-deps',protocol['native_deps'],
                    '--depth-min','.2','--depth-max','.65'],[rep/'BATCH_REPRESENTATION.json',target/'RESULTS.json',
                        target/'raw_camera_candidates.npz',target/'representation_outputs.npz'],protocol)
                common_artifacts['representation_stage']=binding(directory/'stages/representation/STAGE.json')
                if stage['status']!='complete':raise RuntimeError('representation:'+str(stage.get('error')))
                batch_rep=read(rep/'BATCH_REPRESENTATION.json')
                if batch_rep.get('network_forward_count')!=1 or batch_rep.get('hand_head_load_count')!=0 or batch_rep.get('completed_forward_objects')!=1:
                    raise ValueError('Exactly one identity CGR forward and zero scoring heads required per scene')
                domain=directory/'DOMAIN.json'
                if not domain.exists():save(domain,protocol['domain'],exclusive=True)
                def prepare_one(g):
                    prepared=directory/f'g{g}'
                    return _stage(directory/f'stages/prepare_g{g}',[py,'-B',str(HERE/'author_lift_pipeline.py'),'prepare',
                        '--scene',str(capture/'observed_scene.json'),'--observation',str(capture/'observation.json'),
                        '--representation-dir',str(target),'--gesture-id',str(g),'--source-type',str(SOURCE_TYPES[g]),
                        '--seed',str(protocol['queue']['seed']),'--calibration',protocol['inputs']['calibration']['path'],
                        '--cloud-manifest',protocol['inputs']['cloud_manifest']['path'],'--backend',protocol['inputs']['backend']['path'],
                        '--domain',str(domain),'--output-dir',str(prepared)],[prepared/'LIFT_PIPELINE_RESULTS.json'],protocol)
                with ThreadPoolExecutor(max_workers=protocol['prepare_workers'])as executor:
                    prepared_stages=dict(zip(active,executor.map(prepare_one,active)))
                for g in active:  # Frozen order, independent of completion order or outcomes.
                    rows.append(_finish_gesture(batch,protocol,index,row,g,directory,prepared_stages[g],common_artifacts))
            except BaseException as error:
                rows=ledger(batch,protocol)
                recorded={(r['queue_index'],r['gesture_id'])for r in rows}
                for g in active:
                    if(index,g)not in recorded:
                        rows.append(_terminal(batch,protocol,index,row,g,'infrastructure_failure',error=repr(error),artifacts=common_artifacts))
                if isinstance(error,(KeyboardInterrupt,SystemExit,InterruptedChildUnresolved)):raise
            finally:
                verify(protocol['inputs']);verify(protocol['sources'])
                progress=_progress(protocol,rows);save(batch/'PROGRESS.json',progress)
                save(batch/'LEDGER.json',dict(protocol_sha256=sha(batch/'BATCH_PROTOCOL.json'),rows=rows))
        progress=_progress(protocol,rows)
        if len(rows)==4*len(scenes) and not progress['quota_met']:progress['status']='budget_exhausted_insufficient_labels'
        save(batch/'PROGRESS.json',progress)
        return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for key in ('queue','output-dir','calibration','cloud-manifest','backend'):f.add_argument('--'+key,type=Path,required=True)
    for key in ('scene-entry','capture-entry','execution-entry','audit-entry','checkpoint','native-deps','domain'):f.add_argument('--'+key,type=Path)
    f.add_argument('--fit-per-gesture',type=int,default=1000);f.add_argument('--internal-per-gesture',type=int,default=200)
    f.add_argument('--prepare-workers',type=int,default=3);f.add_argument('--stage-timeout-s',type=float,default=3600.)
    r=sub.add_parser('run');r.add_argument('--batch-dir',type=Path,required=True);r.add_argument('--max-scenes',type=int)
    a=parser.parse_args()
    if a.command=='freeze':
        result=freeze(a.queue,a.output_dir,calibration=a.calibration,cloud_manifest=a.cloud_manifest,backend=a.backend,
            fit_per_gesture=a.fit_per_gesture,internal_per_gesture=a.internal_per_gesture,prepare_workers=a.prepare_workers,
            scene_entry=a.scene_entry,capture_entry=a.capture_entry,execution_entry=a.execution_entry,audit_entry=a.audit_entry,checkpoint=a.checkpoint,native_deps=a.native_deps,
            domain=read(a.domain)if a.domain else None,stage_timeout_s=a.stage_timeout_s)
        print(json.dumps({k:result[k]for k in ('schema','quotas','total_scene_budget','total_gesture_attempt_budget')},indent=2))
    else:print(json.dumps(run(a.batch_dir,max_scenes=a.max_scenes),indent=2))


if __name__=='__main__':main()
