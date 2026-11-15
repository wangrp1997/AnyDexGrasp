"""A single owning process/Kit for original captures and verified executions.

The parent freezes ordered jobs and owns the GPU lease. Capture physics always
gets a fresh scene/context. Execution reuses the existing resident controller;
all original capture/command/label code and full journals remain in use.
"""
from __future__ import annotations
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import threading
import time
import traceback
from types import SimpleNamespace

import author_collection_capture as CAPTURE
import author_resident_execution as RESIDENT
import author_verified_execution as STORAGE
import author_collection_batch as B
import author_contact_compressor as W

CAPTURE_PIN='37779a72183281761defa5c3193bdeb8f7cc091a8d5831dc1b4b499b600e3ad2'
RESIDENT_PIN='346aa86a4632f88c4c8c3dc046e60e2a2c4954482ef13bec01a928401c3c4dac'


def capture_template():
    """Recover the exact already-checked storage AST; change one launcher call."""
    if W.sha(CAPTURE.__file__)!=CAPTURE_PIN:raise ValueError('Frozen capture overlay changed')
    tree=ast.parse(Path(CAPTURE.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='capture_function')
    count=0
    for n in ast.walk(fn):
        if isinstance(n,ast.Return)and ast.unparse(n.value)=="scope['capture']":
            n.value=ast.parse('(module,scope)',mode='eval').body;count+=1
    if count!=1:raise ValueError('Original overlay compiler return changed')
    scope=dict(vars(CAPTURE))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__+':template','exec'),scope)
    module,namespace=scope['capture_function']();sites=0
    for n in ast.walk(module):
        if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='AppLauncher':
            n.func=ast.Name(id='resident_launcher',ctx=ast.Load());n.args.insert(0,ast.Name(id='AppLauncher',ctx=ast.Load()));sites+=1
    if sites!=1:raise ValueError('Exactly one capture launcher site required')
    return ast.fix_missing_locations(module),namespace


class DeferredApp:
    def __init__(self,owner):self.owner=owner
    def close(self,*args,**kwargs):
        self.owner.deferred_closes+=1
        if kwargs.get('exit_code',0):self.owner.poisoned=True


class CollectionRuntime:
    def __init__(self,root,sources):
        if W.sha(RESIDENT.__file__)!=RESIDENT_PIN:raise ValueError('Frozen resident source changed')
        self.root=Path(root);self.sources=B.normalized_bindings(sources);B.verify(self.sources)
        self.launcher=None;self.Kit_starts=0;self.context_clears=0;self.deferred_closes=0
        self.closed=self.poisoned=False;self.jobs=[];self.last_role=None;self.owner=threading.get_ident()
        self.execution=RESIDENT.ResidentExecutionSession(self.root/'execution_session',source_bindings=self.sources)
        module,scope=capture_template();scope['resident_launcher']=self.acquire
        exec(compile(module,__file__+':resident_capture','exec'),scope);self.original_capture=scope['capture']

    def acquire(self,launcher_type,**kwargs):
        if threading.get_ident()!=self.owner:raise RuntimeError('Kit must stay on its owning thread')
        if self.launcher is None:
            self.launcher=launcher_type(**kwargs);self.Kit_starts+=1
        return SimpleNamespace(app=DeferredApp(self))

    def clear_scene(self):
        if self.launcher is None:return
        import isaaclab.sim as sim
        current=sim.SimulationContext.instance()
        if current is not None:
            before=current.get_physics_step_count();current.stop()
            if current.get_physics_step_count()!=before:raise RuntimeError('Boundary cleanup integrated physics')
            sim.SimulationContext.clear_instance();sim.create_new_stage();self.context_clears+=1
            self.execution.controller=None;self.execution._geometry=None
            gc.collect()

    def run_job(self,job):
        if self.closed or self.poisoned:raise RuntimeError('Closed/failed worker cannot dispatch')
        if threading.get_ident()!=self.owner:raise RuntimeError('Wrong native worker thread')
        B.verify(self.sources);begin=time.monotonic();row=dict(job=job,status='failed',started_unix_s=time.time())
        try:
            if job['role']=='capture':
                self.clear_scene()
                self.original_capture(Path(job['scene']),Path(job['output_dir']))
                result=B.read(Path(job['output_dir'])/'RESULTS.json')
                if result['status']!='complete':raise RuntimeError('Original capture failed')
                row.update(result=B.binding(Path(job['output_dir'])/'RESULTS.json'),physics_steps=result['physics_ticks'])
            elif job['role']=='execute':
                if self.launcher is None:raise RuntimeError('Diagnostic protocol must launch camera-enabled Kit through capture first')
                ready=B.read(Path(job['prepared_dir'])/'READY.json')
                if B.read(ready['plan']['path'])['total_command_steps']>8000:raise ValueError('Finite command budget exceeded')
                if self.last_role=='capture':self.clear_scene()
                self.execution.launcher=SimpleNamespace(app=DeferredApp(self))
                result=self.execution.execute(Path(job['prepared_dir']))
                row.update(result=B.binding(Path(job['prepared_dir'])/'LIFT_PIPELINE_EXECUTION.json'),physics_steps=result.get('physics_steps'))
                if result.get('status')=='infrastructure_failure':raise RuntimeError('Original execution failed')
            else:raise ValueError('Unknown frozen job role')
            B.verify(self.sources);row['status']='complete';self.last_role=job['role']
        except BaseException:
            self.poisoned=True;row['error']=traceback.format_exc();raise
        finally:
            row.update(wall_s=time.monotonic()-begin,finished_unix_s=time.time());self.jobs.append(row);self.save()
        return row

    def save(self,status=None):
        value=dict(schema='anydex_boya_collection_service_v1',status=status or('failed'if self.poisoned else'open'),
            Kit_starts=self.Kit_starts,explicit_context_clears=self.context_clears,deferred_closes=self.deferred_closes,
            execution_scene_rebuilds=self.execution.scene_rebuild_count,jobs=self.jobs,source_bindings=self.sources,
            diagnostic_only=True,exclude_training=True,new_training_labels=0,retries=0)
        W.save(self.root/'WORKER_RECEIPT.json',value);return value

    def close(self):
        if self.closed:return
        self.closed=True;code=int(self.poisoned)
        # Session close is deferred too. Persist every receipt before real fast shutdown.
        try:
            self.execution.close(exit_code=code);self.save('closed_after_failure'if code else'closed')
        finally:
            if self.launcher is not None:self.launcher.app.close(exit_code=code)


def serve(root,protocol_path,lease_fd):
    import author_collection_wave_resources as R
    root=Path(root).resolve();p=B.read(protocol_path);protocol=B.binding(protocol_path)
    B.verify(p['sources']);B.verify(p['inputs'])
    if(p['diagnostic_only']is not True or p['exclude_training']is not True
       or [j['role']for j in p['worker_jobs']]!=['capture','capture','execute','execute']
       or p['max_execution_steps_per_job']!=8000):raise ValueError('Only frozen four-job diagnostic service')
    lock=B.ROOT/'outputs/author_collection_gpu0.lock'
    R.check_inherited_lease(lease_fd,lock,list((lock.stat().st_dev,lock.stat().st_ino)))
    mailbox=root/'mailbox';mailbox.mkdir(parents=True,exist_ok=False)
    runtime=CollectionRuntime(root,p['sources']);W.save(root/'SERVICE_READY.json',dict(pid=os.getpid(),protocol=protocol))
    error=None;index=0
    try:
        with STORAGE.install():
            while True:
                file=mailbox/f'{index:03d}.json';deadline=time.monotonic()+p['service_idle_timeout_s']
                while not file.exists():
                    if time.monotonic()>deadline:raise TimeoutError('Finite service request deadline')
                    time.sleep(.02)
                request=B.read(file)
                if request['sequence']!=index or request['protocol']!=protocol:raise ValueError('Unbound/duplicate request')
                if request['role']=='abort':
                    runtime.poisoned=True
                    W.save(root/'SERVICE_ABORT.json',dict(sequence=index,request=B.binding(file)))
                    break
                if request['role']=='close':
                    if index!=len(p['worker_jobs'])and not runtime.poisoned:raise ValueError('Early close is only allowed after failure')
                    break
                if index>=len(p['worker_jobs'])or request['job']!=p['worker_jobs'][index]:raise ValueError('Frozen job order/paths changed')
                row=runtime.run_job(request['job'])
                W.save(mailbox/f'{index:03d}.RESULT.json',dict(status='complete',request=B.binding(file),job=row))
                index+=1
    except BaseException:
        error=traceback.format_exc();runtime.poisoned=True
        W.save(root/'SERVICE_ERROR.json',dict(error=error,sequence=index,protocol=protocol))
    finally:runtime.close()
    if error:raise RuntimeError(error)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--lease-fd',type=int,required=True)
    a=parser.parse_args();serve(a.root,a.protocol,a.lease_fd)
