"""Finite new-scene production pilot around the pinned eight-slot runtime.

The diagnostic collector and historical evidence remain immutable. This distinct
schema admits only newly executed, independently certified trials. The fixed
32-scene pilot never starts training or extends its budget to fill a quota.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
from types import SimpleNamespace

import author_pipeline_collection as P

B,R,C=P.B,P.R,P.C
SCHEMA='anydex_boya_author_production_pilot_v1'
CHILD_SCHEMA=SCHEMA+'_window'
PROTOCOL='PRODUCTION_PROTOCOL.json'
CHILD_PROTOCOL='PRODUCTION_WINDOW_PROTOCOL.json'
PIPELINE_SHA='c4a4c0e03c8eac4bada4e4c9f2e722fc1335bb502822ccaf5f984ab326fbbb4b'
WINDOW=8
GOOD={'labelled','null_geometry','no_candidate','placement_null'}


class ProductionStopped(RuntimeError):pass


class StopControl:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.Lock();self.event=threading.Event()
        self.started=time.monotonic()

    def stop(self,reason,**details):
        with self.lock:
            if not self.event.is_set():
                B.save(self.root/'STOP_INTENT.json',dict(reason=reason,unix_s=time.time(),
                    policy='Stop new non-audit stages; drain all already-started processes and original audits',
                    **details),exclusive=True)
                self.event.set()

    def check(self,audit=False):
        if(self.root/'STOP_REQUEST.json').exists():self.stop('External durable stop request',request=B.binding(self.root/'STOP_REQUEST.json'))
        if time.monotonic()-self.started>14400:self.stop('Frozen four-hour controller deadline reached')
        if self.event.is_set() and not audit:raise ProductionStopped('Production dispatch stopped; see STOP_INTENT.json')


def namespace():
    # This pinned adapter already checks every original AST edit and source SHA.
    if B.sha(P.__file__)!=PIPELINE_SHA:raise ValueError('Reviewed eight-slot adapter changed')
    ns=P.build_namespace()
    ns.update(SCHEMA=CHILD_SCHEMA,PROTOCOL=CHILD_PROTOCOL,verify_protocol=verify_window,
        _terminal=terminal,ledger=window_ledger,progress=window_progress)
    return ns


def freeze(source_batch,output):
    """Freeze all 32 new scenes and four windows before any physical work."""
    root=Path(output).resolve();source=Path(source_batch).resolve()/'BATCH_PROTOCOL.json'
    base=B.read(source);queue=P.OLD._base(base)
    if len(queue['scenes'])!=32 or [r['split'] for r in queue['scenes']]!=['fit','internal']*16:
        raise ValueError('Exactly16fit/16internal in predeclared interleaved order required')
    if base['quotas']!=dict(fit=1000,internal=200):raise ValueError('Keep the eventual target separate from this pilot budget')
    root.mkdir(parents=True,exist_ok=False)
    sources=dict(base['sources'])
    for path in [Path(__file__),Path(P.__file__),Path(C.__file__),B.HERE/'run_production_collection.py',
                 B.HERE/'observe_production_collection.py',B.HERE/'audit_author_resident_collection.py']:
        sources[path.name]=B.binding(path)
    plan=dict(schema=SCHEMA,source_batch_protocol=B.binding(source),source_queue=base['inputs']['queue'],
        queue=queue,sources=sources,window_size=WINDOW,scene_budget=32,gesture_slot_budget=128,
        source_scene_indices=list(range(32)),retry_budget=0,automatic_training=False,
        production_scope='Finite new-scene pilot; targets1000fit+200internal per gesture are not a stopping quota in this pilot',
        admission_policy='Original scene/gesture order; certify each executed0/1 once; no class balancing or replacement',
        stop_policy='First infrastructure/integrity error stops subsequent non-audit dispatch and windows; drain in-flight work',
        max_controller_wall_s=14400,resource_limits=dict(kit=8,audit=2,prepare=3,observation=1))
    B.save(root/'PLAN.json',plan,exclusive=True)
    ns=namespace();children=[]
    for w in range(4):
        directory=root/'windows'/f'{w:02d}';indices=list(range(w*WINDOW,(w+1)*WINDOW))
        child=ns['freeze_from_batch'](source.parent,indices,directory,resident_workers=8)
        smi=shutil.which('nvidia-smi')
        if not smi:raise ValueError('GPU0 identity sampler required')
        child.update(schema=CHILD_SCHEMA,diagnostic_only=False,exclude_training=False,
            formal_training_eligible=True,purpose='new_scene_production_pilot_window',
            audit_workers=2,retry_budget=0,execution_grouping='one_original_ready_trial_per_cold_resident_session',
            resource_limits=plan['resource_limits'],gpu_uuid=P.GPU_UUID,
            gpu_identity=C.gpu_identity(smi,P.GPU_UUID),stage_timeout_s=1200.,
            cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,
            production_plan=B.binding(root/'PLAN.json'),window_index=w,
            capture_physics_device='cpu',execution_physics_device='cpu',automatic_training=False,
            no_candidate_fallback=True)
        child['env']=dict(child['env'],CUDA_VISIBLE_DEVICES=P.GPU_UUID,NVIDIA_VISIBLE_DEVICES='0')
        child['inputs'].update(production_plan=child['production_plan'],nvidia_smi=B.binding(smi))
        child['sources'].update(sources)
        child['source_archive']=C.archive_sources(directory,child['sources'])
        B.save(directory/CHILD_PROTOCOL,child)
        B.save(directory/'PROGRESS.json',window_progress(child,[]))
        verify_window(directory)
        children.append(dict(window_index=w,source_scene_indices=indices,protocol=B.binding(directory/CHILD_PROTOCOL)))
    B.save(root/'WINDOW_INDEX.json',dict(plan=B.binding(root/'PLAN.json'),windows=children),exclusive=True)
    protocol=copy.deepcopy(base)
    protocol.update(schema=SCHEMA,plan=B.binding(root/'PLAN.json'),window_index=B.binding(root/'WINDOW_INDEX.json'),
        diagnostic_only=False,exclude_training=False,formal_training_eligible=True,automatic_training=False,
        total_scene_budget=32,total_gesture_attempt_budget=128,resource_limits=plan['resource_limits'],
        sources=sources,gpu_uuid=P.GPU_UUID,retry_budget=0,max_controller_wall_s=14400,
        old435_merged=False,training_updates=0)
    protocol['inputs'].update(production_plan=protocol['plan'],window_index=protocol['window_index'])
    B.save(root/PROTOCOL,protocol,exclusive=True)
    B.save(root/'PROGRESS.json',parent_progress(protocol,[],[]))
    return verify(root)


def verify_window(root):
    root=Path(root).resolve();p=B.read(root/CHILD_PROTOCOL)
    if (p['schema']!=CHILD_SCHEMA or p['task']!=B.TASK or p['diagnostic_only'] is not False or p['exclude_training'] is not False
        or p['formal_training_eligible'] is not True or p['automatic_training'] is not False
        or p['resident_workers']!=8 or p['audit_workers']!=2 or p['prepare_workers']!=3
        or p['resource_limits']!=dict(kit=8,audit=2,prepare=3,observation=1)
        or p['retry_budget']!=0 or p['total_scene_budget']!=WINDOW or p['total_gesture_attempt_budget']!=32
        or p['stage_timeout_s']!=1200. or p['gpu_uuid']!=P.GPU_UUID
        or p['execution_grouping']!='one_original_ready_trial_per_cold_resident_session'):
        raise ValueError('Explicit finite production window policy changed')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'archive':p['source_archive']})
    source=B.read(p['source_batch_protocol']['path']);queue=P.OLD._base(source)
    plan=B.read(p['production_plan']['path']);w=p['window_index'];indices=list(range(w*WINDOW,(w+1)*WINDOW))
    expected_env=dict(source['env'],CUDA_VISIBLE_DEVICES=P.GPU_UUID,NVIDIA_VISIBLE_DEVICES='0')
    if (type(w) is not int or not 0<=w<4 or p['source_scene_indices']!=indices
        or p['queue']['scenes']!=[queue['scenes'][i] for i in indices] or p['queue']['seed']!=queue['seed']
        or plan['queue']!=queue or p['env']!=expected_env or p['gestures']!=list(B.GESTURES)
        or any(p[k]!=source[k] for k in ('domain','native_deps','inherited_env'))):
        raise ValueError('Original new queue, seed, inputs or candidate policy changed')
    archive=B.read(p['source_archive']['path'])
    if {(r['source']['path'],r['source']['sha256']) for r in archive['sources']}!={(r['path'],r['sha256'])for r in p['sources'].values()}:
        raise ValueError('Complete runtime source archive required')
    for row in archive['sources']:B.verify({'source':row['source'],'archive':row['archive']})
    return p


def verify(root):
    root=Path(root).resolve();p=B.read(root/PROTOCOL)
    if (p['schema']!=SCHEMA or p['task']!=B.TASK or p['exclude_training'] is not False or p['diagnostic_only'] is not False
        or p['automatic_training'] is not False or p['total_scene_budget']!=32 or p['total_gesture_attempt_budget']!=128
        or p['formal_training_eligible'] is not True or p['quotas']!=dict(fit=1000,internal=200)
        or p['resource_limits']!=dict(kit=8,audit=2,prepare=3,observation=1) or p['old435_merged'] is not False
        or p['gestures']!=list(B.GESTURES) or p['retry_budget']!=0 or p['max_controller_wall_s']!=14400):
        raise ValueError('Finite production pilot required')
    B.verify(p['inputs']);B.verify(p['sources']);B.verify({'plan':p['plan'],'index':p['window_index']})
    index=B.read(p['window_index']['path']);plan=B.read(p['plan']['path'])
    if index['plan']!=p['plan'] or plan['queue']!=p['queue'] or len(index['windows'])!=4:
        raise ValueError('Frozen production DAG changed')
    for w,entry in enumerate(index['windows']):
        B.verify({'window':entry['protocol']});cp=verify_window(Path(entry['protocol']['path']).parent)
        if cp['window_index']!=w or cp['production_plan']!=p['plan'] or cp['source_scene_indices']!=entry['source_scene_indices']:
            raise ValueError('Window order or production ancestor changed')
    return p


def window_progress(protocol,rows,**kwargs):
    value=P.OLD.progress(protocol,rows,**kwargs)
    value.update(schema=CHILD_SCHEMA,exclude_training=False,formal_training_eligible=True,
        counts_scope='New actual execution labels admitted only after original numeric and independent capture certificates')
    return value


def verify_capture(target,artifacts):
    path=Path(target)/'capture/INDEPENDENT_CAPTURE_AUDIT.json'
    audit=B.read(path)
    if audit['status']!='pass' or audit['physics_ticks']!=750 or audit['all_contact_frames']!=750:
        raise ValueError('Complete independent initialization audit required')
    B.verify(B.normalized_bindings(audit['inputs_sha256']))
    artifacts['independent_capture_audit']=B.binding(path)


def terminal(root,protocol,i,row,g,status,*,artifacts,trial=None,error=None,execution=None,attempted=False):
    verify_window(root);artifacts=dict(artifacts);B.verify(artifacts);execution=execution or {}
    status='labelled' if status=='diagnostic_labelled' else status
    value=dict(schema=CHILD_SCHEMA,task=B.TASK,protocol_sha256=B.sha(Path(root)/CHILD_PROTOCOL),
        queue_index=protocol['source_scene_indices'][i],local_queue_index=i,scene_id=row['scene_id'],
        split=row['split'],gesture_id=g,trial_id=f'{row["scene_id"]}:g{g}',source_object_ids=row['source_object_ids'],
        source_object_digests=row['source_object_digests'],scene_spec=row['scene_spec'],status=status,
        raw_label=None,label=None,exclude_training=False,formal_training_eligible=False,
        artifact_bindings=artifacts,error=error,actual_physics_steps=execution.get('physics_steps',0),
        execution_steps_unknown=bool(attempted and(execution.get('status')in(None,'starting')or execution.get('_execution_steps_unknown',False))))
    if trial is not None:
        prepared=Path(trial['record']['path']).parent
        verified=B.validate_trial(prepared,row)
        if verified!=trial or trial['gesture_id']!=g or status!='labelled':raise ValueError('Actual trial changed before admission')
        B.certify_audit(B.read(artifacts['runtime_audit']['path']),trial,row)
        verify_capture(prepared.parent,artifacts)
        value.update(raw_label=trial['label'],label=trial['label'],trial=trial,formal_training_eligible=True,
            actual_physics_steps=trial['actual_physics_steps'],execution_steps_unknown=False)
        artifacts.update({k:trial[k]for k in('record','candidate','evaluation','commands','command_binding','scene')})
    if trial is None and status=='labelled':raise ValueError('No actual trial may become a label')
    B.verify(artifacts)
    B.save(Path(root)/'ledger'/f'{value["queue_index"]:07d}_g{g}.json',value,exclusive=True)
    return value


def window_ledger(root):
    root=Path(root);p=verify_window(root);rows=[]
    for path in sorted((root/'ledger').glob('*.json')):
        row=B.read(path)
        if row['protocol_sha256']!=B.sha(root/CHILD_PROTOCOL) or row['exclude_training'] is not False:
            raise ValueError('Production ledger protocol changed')
        if (row['label'] is not None)!=(row['status']=='labelled' and row['formal_training_eligible'] is True):
            raise ValueError('Uncertified production label')
        B.verify(row['artifact_bindings'])
        for entry in row['artifact_bindings'].values():
            if Path(entry['path']).name=='STAGE.json':B.verify(B.read(entry['path']).get('outputs',{}))
        rows.append(row)
    return rows


@contextmanager
def guarded_stages(control):
    direct=B._stage
    def stage(directory,command,expected,protocol):
        is_audit='audit_author_collection_runtime.py' in [Path(str(x)).name for x in command]
        control.check(audit=is_audit)
        try:
            result=direct(directory,command,expected,protocol)
            if result['status']!='complete':control.stop('External stage failed',stage_dir=str(directory))
            elif Path(directory).name=='capture':
                capture=Path(expected[0]).parent;output=capture/'INDEPENDENT_CAPTURE_AUDIT.json'
                ap=copy.deepcopy(protocol);ap['sources']['audit_entry']=protocol['sources']['audit_author_resident_collection.py']
                auditdir=Path(directory).with_name('capture_audit')
                audited=direct(auditdir,[protocol['inputs']['python']['path'],'-B',ap['sources']['audit_entry']['path'],
                    'capture','--capture-dir',str(capture),'--output',str(output)],[output],ap)
                if audited['status']!='complete' or B.read(output).get('status')!='pass':
                    control.stop('Independent capture audit failed',stage_dir=str(auditdir));raise ProductionStopped('Capture audit failed')
            if result['status']=='complete' and Path(directory).name.startswith('prepare_g'):
                if B.read(expected[0]).get('status')=='infrastructure_failure':control.stop('Preparation infrastructure failure',stage_dir=str(directory))
            return result
        except BaseException as error:
            control.stop('Stage or integrity exception',stage_dir=str(directory),error=repr(error));raise
    B._stage=stage
    try:yield
    finally:B._stage=direct


def parent_progress(protocol,rows,windows,*,started=None,finished=None,error=None):
    counts={s:{str(g):0 for g in B.GESTURES}for s in('fit','internal')};positive=copy.deepcopy(counts)
    for row in rows:
        if row['label'] is not None:
            counts[row['split']][str(row['gesture_id'])]+=1;positive[row['split']][str(row['gesture_id'])]+=row['label']
    # Count capture cost from durable runtime receipts even if integrity prevents
    # ledger admission. Interrupted capture starts stay explicitly unknown.
    root=Path(protocol['plan']['path']).parent;capture_ticks=0;capture_unknown=0;capture_attempts=0
    for stage in root.glob('windows/*/scenes/*/stages/capture/STAGE.json'):
        capture_attempts+=1;capture=stage.parents[2]/'capture'
        receipt=capture/'RESULTS.json';progress=capture/'PROGRESS.json'
        value=B.read(receipt)if receipt.exists()else B.read(progress)if progress.exists()else{}
        capture_ticks+=int(value.get('physics_ticks',0))
        capture_unknown+=value.get('status')!='complete'
    return dict(schema=SCHEMA,task=B.TASK,status='complete'if len(rows)==128 and not error else 'stopped'if error else 'running'if started else 'frozen',
        total_scene_budget=32,total_gesture_attempt_budget=128,terminal_slots=len(rows),eligible_labels=counts,positive_labels=positive,
        status_counts={s:sum(r['status']==s for r in rows)for s in sorted({r['status']for r in rows})},
        known_final_execution_physics_steps=sum(w['known_final_execution_physics_steps']for w in windows),
        execution_starts_with_unknown_steps=sum(w['execution_starts_with_unknown_steps']for w in windows),
        captured_initialization_ticks=capture_ticks,captured_initialization_object_steps=3*capture_ticks,
        capture_attempts=capture_attempts,capture_starts_with_unknown_steps=capture_unknown,
        windows_finished=len(windows),started_unix_s=started,finished_unix_s=finished,error=error,
        exclude_training=False,old435_merged=False,training_updates=0,new_model_count=0,automatic_training=False,
        budget_exhaustion_is_target_completion=False,target_fit_per_gesture=1000,target_internal_per_gesture=200)


def admit_window(root,protocol,entry):
    directory=Path(entry['protocol']['path']).parent;rows=window_ledger(directory);admitted=[]
    for raw in rows:
        i,g=raw['queue_index'],raw['gesture_id'];spec=protocol['queue']['scenes'][i]
        if raw['scene_id']!=spec['scene_id'] or raw['source_object_digests']!=spec['source_object_digests'] or raw['split']!=spec['split']:
            raise ValueError('Parent full-scene identity or split differs')
        if raw['label'] is not None:
            trial=B.validate_trial(Path(raw['trial']['record']['path']).parent,spec)
            if trial!=raw['trial']:raise ValueError('Parent trial differs')
            B.certify_audit(B.read(raw['artifact_bindings']['runtime_audit']['path']),trial,spec)
        value=dict(raw,schema=SCHEMA,protocol_sha256=B.sha(root/PROTOCOL),window_protocol=entry['protocol'],
            raw_ledger=B.binding(directory/'ledger'/f'{i:07d}_g{g}.json'))
        B.save(root/'ledger'/f'{i:07d}_g{g}.json',value,exclusive=True);admitted.append(value)
    return admitted


def ledger(root,protocol=None):
    root=Path(root);protocol=protocol or verify(root);rows=[];seen=set()
    entries=B.read(protocol['window_index']['path'])['windows']
    for path in sorted((root/'ledger').glob('*.json')):
        row=B.read(path);key=(row['queue_index'],row['gesture_id'])
        i,g=key
        if(type(i)is not int or not 0<=i<32 or g not in B.GESTURES or key in seen
            or row['protocol_sha256']!=B.sha(root/PROTOCOL)):raise ValueError('Duplicate or unbound parent outcome')
        entry=entries[i//WINDOW];spec=protocol['queue']['scenes'][i]
        expected=Path(entry['protocol']['path']).parent/'ledger'/f'{i:07d}_g{g}.json'
        if(row['window_protocol']!=entry['protocol']or Path(row['raw_ledger']['path']).resolve()!=expected.resolve()
            or row['scene_id']!=spec['scene_id']or row['split']!=spec['split']
            or row['source_object_ids']!=spec['source_object_ids']or row['source_object_digests']!=spec['source_object_digests']
            or row['trial_id']!=f'{spec["scene_id"]}:g{g}'):
            raise ValueError('Production outcome belongs to another window or scene')
        seen.add(key);B.verify({'raw':row['raw_ledger']});raw=B.read(row['raw_ledger']['path'])
        if(raw['schema']!=CHILD_SCHEMA or raw['protocol_sha256']!=entry['protocol']['sha256']or raw['exclude_training']is not False
            or(row['label']is not None)!=(row['status']=='labelled'and row['formal_training_eligible']is True)):
            raise ValueError('A diagnostic or uncertified raw record cannot enter production')
        if {k:v for k,v in row.items()if k not in('schema','protocol_sha256','window_protocol','raw_ledger')}!={k:v for k,v in raw.items()if k not in('schema','protocol_sha256')}:
            raise ValueError('Parent label, cost or artifact differs from original certified child')
        B.verify(row['artifact_bindings']);rows.append(row)
    return rows


def run(root):
    root=Path(root).resolve();protocol=verify(root);control=StopControl(root);rows=[];windows=[];error=None
    start=time.monotonic();started=time.time()
    with R.lease(root/'DRIVER.lock'),R.lease(P.OLD.LEGACY_LOCK)as fd:
        B.save(root/'RUN_STARTED.json',dict(protocol=B.binding(root/PROTOCOL),process=R.process_identity(os.getpid()),
            started_unix_s=started,no_restart=True),exclusive=True)
        with R.active_run(root,P.OLD.REGISTRY,B.binding(root/PROTOCOL)):
            try:
                for entry in B.read(protocol['window_index']['path'])['windows']:
                    control.check();verify(root);directory=Path(entry['protocol']['path']).parent;cp=verify_window(directory)
                    if C.gpu_identity(cp['inputs']['nvidia_smi']['path'],P.GPU_UUID)!=cp['gpu_identity']:raise ValueError('GPU0 identity changed')
                    pool=P.ExpandedPool(root/'resource_slots',fd,B.binding(directory/CHILD_PROTOCOL));previous={k:os.environ.get(k)for k in cp['env']}
                    os.environ.update(cp['env'])
                    try:
                        with R.install_worker_adapters(pool,P.OLD.LEGACY_LOCK),guarded_stages(control):
                            try:result=namespace()['_run_locked'](directory,cp,pool)
                            finally:
                                saved=directory/'RESIDENT_COLLECTION_RESULTS.json'
                                if saved.exists():windows.append(B.read(saved))
                    finally:
                        for k,v in previous.items():
                            if v is None:os.environ.pop(k,None)
                            else:os.environ[k]=v
                    rows.extend(admit_window(root,protocol,entry))
                    B.save(root/'PROGRESS.json',parent_progress(protocol,rows,windows,started=started))
                    if result.get('error') or any(r['status']not in GOOD for r in rows):control.stop('Window contains infrastructure or incomplete outcomes')
                    control.check()
            except BaseException:
                error=traceback.format_exc();control.stop('Controller stopped',error=error)
                # Reconcile already-written certified rows without restarting any stage.
                for entry in B.read(protocol['window_index']['path'])['windows']:
                    directory=Path(entry['protocol']['path']).parent
                    if any((directory/'ledger').glob('*.json')) and not any(i in {r['queue_index']for r in rows}for i in entry['source_scene_indices']):
                        try:rows.extend(admit_window(root,protocol,entry))
                        except Exception:pass  # Raw files/cost receipts remain authoritative; never invent admission.
            finally:
                result=parent_progress(protocol,rows,windows,started=started,finished=time.time(),error=error)
                result.update(controller_wall_s=time.monotonic()-start,parent_Kit_launches=0,
                    window_results=[B.binding(Path(e['protocol']['path']).parent/'RESIDENT_COLLECTION_RESULTS.json')
                        for e in B.read(protocol['window_index']['path'])['windows'] if(Path(e['protocol']['path']).parent/'RESIDENT_COLLECTION_RESULTS.json').exists()])
                B.save(root/'PROGRESS.json',result);B.save(root/'PRODUCTION_RESULTS.json',result,exclusive=True)
    return result


def export(root,output):
    """Reuse the original array writer with the saved first audit certificates."""
    import author_collection_export as E
    root=Path(root).resolve();protocol=verify(root)
    admitted=ledger(root,protocol);by_record={r['trial']['record']['path']:r for r in admitted if r['label'] is not None}
    if len(by_record)!=sum(r['label']is not None for r in admitted):raise ValueError('Duplicate actual record path')
    tree=ast.parse(Path(E.__file__).read_text());changes=[]
    class Edit(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='BATCH_PROTOCOL.json':changes.append(node.lineno);return ast.copy_location(ast.Constant(PROTOCOL),node)
            return node
    tree=Edit().visit(tree);assert len(changes)==2
    ns=dict(__name__='_anydex_production_export',__file__=E.__file__)
    exec(compile(ast.fix_missing_locations(tree),E.__file__,'exec'),ns)
    proxy=SimpleNamespace(**{k:getattr(B,k)for k in dir(B)});proxy.SCHEMA=SCHEMA;proxy.ledger=ledger
    ns['batch']=proxy
    def certificate(prepared,child_protocol):
        path=Path(prepared)/'RUNTIME_AUDIT.json'
        row=by_record.get(str(Path(prepared)/'COLLECTION_RECORD.json'))
        if row is None:raise ValueError('Saved first audit must belong to one new admitted trial')
        B.verify({'audit':row['artifact_bindings']['runtime_audit']});return B.read(path)
    ns['_audit_trial']=certificate
    return ns['export'](root,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--source-batch',type=Path,required=True);f.add_argument('--output-dir',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--run-dir',type=Path,required=True)
    e=sub.add_parser('export');e.add_argument('--run-dir',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    a=parser.parse_args()
    result=freeze(a.source_batch,a.output_dir)if a.action=='freeze'else run(a.run_dir)if a.action=='run'else export(a.run_dir,a.output_dir)
    print({k:result[k]for k in('status','eligible_labels','terminal_slots','datasets')if k in result})
    if a.action=='run' and result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
