"""Independent eight-slot collection adapter with two first-audit workers.

Compile the pinned resident collector into a private namespace. Only scheduling,
per-trial session paths, diagnostic schema and declared limits change. Original
capture, CGR, prepare, execution, audits and ordered admission remain intact.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import author_resident_collection as OLD
import author_scaling_control as C

B,R,G=OLD.B,OLD.R,C.G
SCHEMA='anydex_boya_author_pipeline_collection_v1'
PROTOCOL='PIPELINE_COLLECTION_PROTOCOL.json'
OLD_SHA='6381790f3f246be9e33b1d8d8d89dc9cfef9acc9128357e8e302b718cf65ad65'
RESOURCE_SHA='8d642f44ddcf085556807739ad27b97e2c8a621e1e61abc1d9c535b08ef7e80b'
GPU_UUID='GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86'


class GracefulProcess:
    def __init__(self,process,directory,protocol):
        self.process,self.directory,self.protocol=process,Path(directory),protocol
        self.identity=R.process_identity(process.pid)

    def __getattr__(self,name):return getattr(self.process,name)

    def wait(self,*args,**kwargs):
        try:return self.process.wait(*args,**kwargs)
        except BaseException as error:
            if self.process.poll() is None:
                path=self.directory/'CANCELLATION.json'
                B.save(path,dict(reason=type(error).__name__,process=self.identity,
                    intended_first_signal='SIGINT',unix_s=time.time()),exclusive=True)
                events=C.finish_owned(self.process,self.identity,self.protocol)
                B.save(path.with_name('CANCELLATION_RESULTS.json'),dict(events=events,
                    returncode=self.process.poll(),process=self.identity))
            raise


class ExpandedPool(R.ResourcePool):
    def __init__(self,root,lease_fd,protocol_binding):
        super().__init__(root,lease_fd,protocol_binding)
        self.pipeline_protocol=B.read(protocol_binding['path'])

    def _acquire(self,kind,reservation):
        if kind=='kit':return C.ScalingPool._acquire(self,kind,reservation)
        return super()._acquire(kind,reservation)

    def popen(self,*args,**kwargs):
        process=super().popen(*args,**kwargs)
        directory=Path(kwargs['stdout'].name).parent
        return GracefulProcess(process,directory,self.pipeline_protocol)


class AuditPipeline:
    def __init__(self,root,protocol):
        self.root,self.protocol=Path(root),protocol
        self.executor=ThreadPoolExecutor(max_workers=protocol['audit_workers'])
        self.lock=threading.Lock();self.jobs={};self.events=[];self.active=0;self.peak=0

    @staticmethod
    def signature(command,expected):
        return tuple(map(str,command)),tuple(str(Path(p).resolve()) for p in expected)

    def queue(self,items,direct,py,protocol):
        for i,row,g,target,directory,_ in items:
            target=Path(target);record=target/'COLLECTION_RECORD.json'
            if not record.is_file() or B.read(record).get('label') is None:continue
            stage=directory/f'stages/audit_g{g}';output=target/'RUNTIME_AUDIT.json'
            command=[py,'-B',protocol['sources']['audit_entry']['path'],'--prepared-dir',str(target),
                '--split-manifest',protocol['queue']['split_manifest']['path'],'--output',str(output)]
            key=str(stage.resolve());signature=self.signature(command,[output])
            with self.lock:
                if key in self.jobs:raise ValueError('An original audit was queued twice')
                event=dict(local_queue_index=i,gesture_id=g,stage_dir=key,queued_unix_s=time.time(),
                    command=command,expected=[str(output)],status='queued')
                B.save(self.root/'audit_queue'/f'{i:07d}_g{g}.json',event,exclusive=True)
                future=self.executor.submit(self._run,event,direct,stage,command,[output])
                self.jobs[key]=(signature,future);self.events.append(event)

    def _run(self,event,direct,directory,command,expected):
        with self.lock:
            self.active+=1;self.peak=max(self.peak,self.active)
            event.update(started_unix_s=time.time(),status='running')
        try:
            result=direct('audit',directory,command,expected)
            event['status']=result['status']
            event['stage']=B.binding(directory/'STAGE.json')
            return result
        except BaseException as error:
            event.update(status='failed',error=repr(error));raise
        finally:
            with self.lock:
                event['finished_unix_s']=time.time();self.active-=1
                B.save(self.root/'audit_queue'/f'{event["local_queue_index"]:07d}_g{event["gesture_id"]}.json',event)

    def stage(self,direct,kind,directory,command,expected,stage_protocol=None):
        if kind=='audit':
            with self.lock:entry=self.jobs.get(str(Path(directory).resolve()))
            if entry is not None:
                C.need(entry[0]==self.signature(command,expected),'Queued original audit command changed')
                return entry[1].result()
        return direct(kind,directory,command,expected,stage_protocol)

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=False)
        C.need(self.active==0 and self.peak<=self.protocol['audit_workers'],'Audit lifecycle exceeded frozen limit')
        B.save(self.root/'AUDIT_PIPELINE_RESULTS.json',dict(schema=SCHEMA,
            audits=self.events,peak_active_audits=self.peak,queued_audits=len(self.events),
            all_futures_terminal=all(f.done() for _,f in self.jobs.values()),
            numeric_audits_reused_from_history=0,new_physics_steps=0),exclusive=True)


def build_namespace():
    """Exact source-pinned scheduling edits; preserve original stage semantics."""
    C.need(B.sha(OLD.__file__)==OLD_SHA and B.sha(R.__file__)==RESOURCE_SHA,
           'Reviewed collector/resource source changed')
    tree=ast.parse(Path(OLD.__file__).read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    edits={}

    def replace_expressions(node,mapping):
        expected={ast.dump(ast.parse(a,mode='eval').body,include_attributes=False):(a,b) for a,b in mapping.items()}
        class Edit(ast.NodeTransformer):
            def visit(self,value):
                if isinstance(value,ast.expr):
                    hit=expected.get(ast.dump(value,include_attributes=False))
                    if hit:
                        a,b=hit;edits[a]=edits.get(a,0)+1
                        return ast.copy_location(ast.parse(b,mode='eval').body,value)
                return super().visit(value)
        return Edit().visit(node)

    for name,condition in (('_freeze','1 <= resident_workers <= 3'),
                           ('verify_protocol',"1 <= protocol['resident_workers'] <= 3")):
        replace_expressions(functions[name],{condition:condition[:-1]+'8'})
    verify_fn=functions['verify_protocol']
    replace_expressions(verify_fn,{"('domain', 'native_deps', 'env', 'inherited_env', 'stage_timeout_s')":
        "('domain', 'native_deps', 'inherited_env')"})
    run_fn=functions['_run_locked']
    direct=next(n for n in run_fn.body if isinstance(n,ast.FunctionDef) and n.name=='stage')
    direct.name='_stage_direct'
    location=run_fn.body.index(direct)+1
    wrapper=ast.parse('''
def stage(kind,directory,command,expected,stage_protocol=None):
    return audit_pipeline.stage(_stage_direct,kind,directory,command,expected,stage_protocol)
''').body[0]
    run_fn.body.insert(location,wrapper)
    run_fn.body.insert(0,ast.parse('audit_pipeline = AuditPipeline(root,protocol)').body[0])
    main_try=next(n for n in run_fn.body if isinstance(n,ast.Try))
    main_try.finalbody.insert(0,ast.parse('audit_pipeline.close()').body[0])
    group_fn=next(n for n in ast.walk(run_fn) if isinstance(n,ast.FunctionDef) and n.name=='execute_group')
    replace_expressions(group_fn,{
        "directory / 'resident_session'":"items[0][3] / 'resident_session'",
        "directory / 'RESIDENT_SOURCE_BINDINGS.json'":"items[0][3] / 'RESIDENT_SOURCE_BINDINGS.json'",
        "directory / 'stages/resident_execute'":"items[0][3] / 'stages/resident_execute'"})
    C.need(ast.unparse(group_fn.body[-1])=='return (i, result)','Original execution return changed')
    group_fn.body.insert(-1,ast.parse('audit_pipeline.queue(items,_stage_direct,py,protocol)').body[0])
    group_fn.body[-1]=ast.parse('return (items[0][0],items[0][2]), result').body[0]
    replace_expressions(run_fn,{
        'groups.setdefault(item[0], [])':'groups.setdefault((item[0],item[2]), [])',
        "directory / 'stages/resident_execute/STAGE.json'":"target / 'stages/resident_execute/STAGE.json'",
        "directory / 'resident_session/EXECUTE_MANY_RESULTS.json'":"target / 'resident_session/EXECUTE_MANY_RESULTS.json'",
        "directory / 'resident_session/SESSION_RECEIPT.json'":"target / 'resident_session/SESSION_RECEIPT.json'",
        'group_results[i]':'group_results[(i,g)]'})
    replace_expressions(functions['run'],{'R.ResourcePool(root / \'resource_slots\', lease_fd, B.binding(root / PROTOCOL))':
        'ExpandedPool(root / \'resource_slots\', lease_fd, B.binding(root / PROTOCOL))'})
    expected={
        '1 <= resident_workers <= 3':1,"1 <= protocol['resident_workers'] <= 3":1,
        "('domain', 'native_deps', 'env', 'inherited_env', 'stage_timeout_s')":1,
        "directory / 'resident_session'":1,"directory / 'RESIDENT_SOURCE_BINDINGS.json'":1,
        "directory / 'stages/resident_execute'":1,'groups.setdefault(item[0], [])':1,
        "directory / 'stages/resident_execute/STAGE.json'":2,
        "directory / 'resident_session/EXECUTE_MANY_RESULTS.json'":1,
        "directory / 'resident_session/SESSION_RECEIPT.json'":1,'group_results[i]':1,
        "R.ResourcePool(root / 'resource_slots', lease_fd, B.binding(root / PROTOCOL))":1}
    C.need(edits==expected,'Original scheduling AST differs from reviewed edit sites: '+repr(edits))
    ns=dict(__name__='_anydex_pipeline_collection',__file__=str(Path(OLD.__file__).resolve()),
        ExpandedPool=ExpandedPool,AuditPipeline=AuditPipeline)
    exec(compile(ast.fix_missing_locations(tree),str(OLD.__file__),'exec'),ns)
    ns.update(SCHEMA=SCHEMA,PROTOCOL=PROTOCOL)
    ns['_base_verify']=ns['verify_protocol']
    ns['verify_protocol']=lambda root:verify(root,_ns=ns)
    ns['ast_adaptations']=dict(expression_sites=edits,extra_hooks=[
        'private AuditPipeline lifecycle','original audit stage future handoff',
        'queue after each complete execution subprocess','drain before final receipts'])
    return ns


def freeze(source_batch,indices,root,*,workers=8):
    C.need(type(workers) is int and 1<=workers<=8 and 1<=len(indices)<=2,
           'This diagnostic permits one/two scenes and at most eight workers')
    root=Path(root).resolve();ns=build_namespace()
    protocol=ns['freeze_from_batch'](source_batch,indices,root,resident_workers=workers)
    smi=shutil.which('nvidia-smi');C.need(smi is not None,'GPU0 identity sampler required')
    protocol.update(audit_workers=2,retry_budget=0,execution_grouping='one_original_ready_trial_per_cold_resident_session',
        execution_policy='One original first-candidate trial per subprocess; original fixed ledger order',
        audit_policy='Queue each original first numeric audit after its execution process exits; max2; ordered ledger after certificate',
        resource_limits=dict(kit=8,audit=2,prepare=3,observation=1),
        gpu_uuid=GPU_UUID,gpu_identity=C.gpu_identity(smi,GPU_UUID),
        stage_timeout_s=1200.,cancel_sigint_grace_s=30.,cancel_sigterm_grace_s=5.,cancel_sigkill_grace_s=5.,
        total_capture_attempt_budget=len(indices),total_execution_attempt_budget=4*len(indices),
        initial_tick_budget=750*len(indices),capture_physics_device='cpu',execution_physics_device='cpu',
        no_candidate_fallback=True,formal_training_eligible=False)
    protocol['env']=dict(protocol['env'],CUDA_VISIBLE_DEVICES=GPU_UUID,NVIDIA_VISIBLE_DEVICES='0')
    protocol['inputs']['nvidia_smi']=B.binding(smi)
    protocol['inputs']['cpu_capacity_evidence']=B.binding(B.HERE/'evidence/author_cpu_scaling_20261115_v1/EVIDENCE.json')
    for path in (Path(__file__),Path(C.__file__),B.HERE/'run_pipeline_collection.py'):
        protocol['sources'][path.name]=B.binding(path)
    protocol['source_archive']=C.archive_sources(root,protocol['sources'])
    B.save(root/PROTOCOL,protocol)
    B.save(root/'PIPELINE_ADAPTER.json',dict(original_source=B.binding(OLD.__file__),
        adapter=B.binding(__file__),changes=ns['ast_adaptations']),exclusive=True)
    B.save(root/'PROGRESS.json',ns['progress'](protocol,[]))
    return verify(root,_ns=ns)


def verify(root,*,_ns=None):
    ns=build_namespace() if _ns is None else _ns
    root=Path(root).resolve();protocol=ns['_base_verify'](root)
    C.need(protocol['audit_workers']==2 and protocol['resource_limits']==dict(kit=8,audit=2,prepare=3,observation=1)
        and protocol['execution_grouping']=='one_original_ready_trial_per_cold_resident_session'
        and protocol['stage_timeout_s']==1200. and protocol['total_scene_budget']<=2
        and protocol['total_execution_attempt_budget']==4*protocol['total_scene_budget']
        and protocol['total_capture_attempt_budget']==protocol['total_scene_budget']
        and protocol['retry_budget']==0 and protocol['formal_training_eligible'] is False,
        'Explicit finite pipeline diagnostic policy changed')
    source=B.read(protocol['source_batch_protocol']['path'])
    expected=dict(source['env'],CUDA_VISIBLE_DEVICES=GPU_UUID,NVIDIA_VISIBLE_DEVICES='0')
    C.need(protocol['env']==expected and protocol['gpu_uuid']==GPU_UUID,
           'Only explicit physical GPU0 UUID remapping is allowed')
    B.verify({'archive':protocol['source_archive']})
    archive=B.read(protocol['source_archive']['path'])
    C.need({(r['source']['path'],r['source']['sha256']) for r in archive['sources']}==
        {(s['path'],s['sha256']) for s in protocol['sources'].values()},'Complete runtime archive required')
    for row in archive['sources']:B.verify({'original':row['source'],'archive':row['archive']})
    return protocol


def run(root):
    root=Path(root).resolve();ns=build_namespace();protocol=verify(root,_ns=ns)
    C.need(C.gpu_identity(protocol['inputs']['nvidia_smi']['path'],GPU_UUID)==protocol['gpu_identity'],
           'Frozen GPU0 identity changed')
    began=time.monotonic();result=ns['run'](root)
    rows=ns['ledger'](root)
    clean=all(r['status'] in ('diagnostic_labelled','null_geometry','no_candidate','placement_null') for r in rows)
    result=dict(result,pipeline=B.binding(root/'AUDIT_PIPELINE_RESULTS.json'),
        original_collector_source=B.binding(OLD.__file__),adapter=B.binding(__file__),
        execution_grouping=protocol['execution_grouping'],run_api_wall_s=time.monotonic()-began,
        pipeline_validity='pass' if clean and not result.get('error') else 'failed',
        ledger_statuses=[r['status'] for r in rows])
    B.save(root/'PIPELINE_COLLECTION_RESULTS.json',result,exclusive=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--source-batch',type=Path,required=True)
    f.add_argument('--scene-indices',nargs='+',type=int,required=True);f.add_argument('--workers',type=int,default=8)
    f.add_argument('--output-dir',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    result=freeze(args.source_batch,args.scene_indices,args.output_dir,workers=args.workers) if args.action=='freeze' else run(args.run_dir)
    print(json.dumps(dict(status=result.get('status','frozen'),slots=result.get('total_gesture_attempt_budget'),
        known_execution_steps=result.get('known_final_execution_physics_steps')),indent=2))
    if args.action=='run' and (result.get('status')!='complete' or result.get('pipeline_validity')!='pass'):raise SystemExit(1)


if __name__=='__main__':main()
