"""Once-only native lift execution with a complete lossless contact journal.

Only contact storage and receipt finalization are adapted. The frozen runtime's
step loop, physical control, contact measurements and finish AST remain intact.
No simulator is imported until the existing pipeline explicitly launches it.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LIFT_SHA256 = '40e519d035c45bd25289fe5dcb154808baef1153e82c62e0cf69b5110b47b20d'
POINTER_SCHEMA = 'anydex_boya_contact_journal_pointer_v1'
ENTRY_SCHEMA = 'anydex_boya_streamed_lift_execution_entry_v1'
LAUNCHER_AST_SHA256 = '227d0e3ca6c9b3d97c3e1c9831e991ecadda8438c7db63c8fe3a8b325c4fc01e'
GPU0_KIT_ARGS = '--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


class ContactJournal:
    """Append one complete original contact row per tick, with bounded memory."""
    def __init__(self,path,*,source_bindings=None):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        self.raw=self.path.open('xb')
        self.stream=gzip.GzipFile(filename='',fileobj=self.raw,mode='wb',compresslevel=1,mtime=0)
        self.records=0;self.closed=False;self.source_bindings=dict(source_bindings or {})

    def append(self,row):
        if self.closed: raise ValueError('Contact journal is already finalized')
        if not isinstance(row,dict) or row.get('step') != self.records+1:
            raise ValueError('One ordered complete contact record per native tick required')
        data=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        self.stream.write(data);self.records+=1

    def close(self):
        if not self.closed:
            try:self.stream.close()
            finally:self.raw.close();self.closed=True

    def pointer(self):
        self.close()
        return dict(schema=POINTER_SCHEMA,compression='gzip',record_format='jsonl',
            record_count=self.records,data=dict(path=str(self.path),sha256=sha(self.path),bytes=self.path.stat().st_size),
            source_bindings=self.source_bindings,lossless=True,contact_records_omitted=0,
            semantics='Each line is one complete original runtime contact row; this document is an explicit pointer, not a rows array')


class _SaveProxy:
    def __init__(self,common):self.common=common
    def __getattr__(self,name):return getattr(self.common,name)
    def save(self,path,value):
        if (isinstance(value,dict) and set(value)=={'rows'}
                and isinstance(value['rows'],ContactJournal)):
            return self.common.save(path,value['rows'].pointer())
        return self.common.save(path,value)


def runtime_class(lift,source_bindings,journals):
    """Compile the exact frozen finish AST with only its save dependency wrapped."""
    if sha(lift.__file__) != LIFT_SHA256:
        raise ValueError('Frozen original lift source changed')
    tree=ast.parse(Path(lift.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiObjectLiftRuntime')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='finish')
    namespace=dict(vars(lift));namespace['common']=_SaveProxy(lift.common)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(lift.__file__),'exec'),namespace)
    original_finish=namespace['finish']
    ast_sha=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
    original_class=lift.MultiObjectLiftRuntime
    class StreamedRuntime(original_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.contact_rows=ContactJournal(self.output_dir/'MULTI_OBJECT_CONTACTS.jsonl.gz',source_bindings=source_bindings)
            journals.append(self.contact_rows)
        def finish(self,*args,**kwargs):
            try:return original_finish(self,*args,**kwargs)
            finally:
                self.contact_rows.close()
                # Preserve complete contact data even if the original evaluator
                # itself raises before reaching its normal contact save.
                pointer_path=self.output_dir/'MULTI_OBJECT_CONTACTS.json'
                if not pointer_path.exists():lift.common.save(pointer_path,self.contact_rows.pointer())
    return StreamedRuntime,ast_sha


def _modules():
    if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
    return importlib.import_module('author_lift'),importlib.import_module('author_lift_pipeline')


def _launch_native_gpu0(common):
    """Original CPU/headless launcher; explicitly restrict Vulkan to GPU0."""
    tree=ast.parse(Path(common.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_launch_native')
    if hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()!=LAUNCHER_AST_SHA256:
        raise ValueError('Original native launcher AST changed')
    changed=0
    for node in ast.walk(fn):
        if isinstance(node,ast.keyword) and node.arg=='kit_args':
            if not isinstance(node.value,ast.Constant) or node.value.value!='--/log/file=/dev/null':
                raise ValueError('Original native launcher kit args changed')
            node.value=ast.Constant(GPU0_KIT_ARGS);changed+=1
    if changed!=1:raise ValueError('Exactly one original launcher kit_args required')
    namespace=dict(vars(common))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(common.__file__),'exec'),namespace)
    return namespace['_launch_native']()


def execute(prepared_dir):
    """CLI entry; the unchanged pipeline still owns its single-attempt guard."""
    prepared_dir=Path(prepared_dir).resolve()
    lift,pipeline=_modules()
    if pipeline._backend() is not lift:raise ValueError('Pipeline/backend module identity differs')
    sources={str(Path(__file__).resolve()):sha(__file__),str(Path(lift.__file__).resolve()):sha(lift.__file__),
             str(HERE/'author_lift_pipeline.py'):sha(HERE/'author_lift_pipeline.py'),
             str(HERE/'author_pipeline.py'):sha(HERE/'author_pipeline.py')}
    journals=[];runtime,ast_sha=runtime_class(lift,sources,journals)
    receipt_path=prepared_dir/'STREAMED_EXECUTION_ENTRY.json'
    if receipt_path.exists():raise FileExistsError('Fresh streamed execution entry receipt required')
    receipt=dict(schema=ENTRY_SCHEMA,status='configured',source_sha256=sources,
        original_finish_ast_sha256=ast_sha,contact_storage_schema=POINTER_SCHEMA,
        original_launcher_ast_sha256=LAUNCHER_AST_SHA256,renderer_active_gpu=0,
        renderer_multi_gpu_enabled=False,physics_device='cpu',
        original_physics_control_and_finish_ast_unchanged=True,contact_records_omitted=0)
    with receipt_path.open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    original_runtime=lift.MultiObjectLiftRuntime;original_launch=pipeline.common._launch_native
    finalized=False
    def finalize():
        nonlocal finalized
        if finalized:return
        for journal in journals:journal.close()
        changed=[path for path,digest in sources.items() if sha(path)!=digest]
        pipeline_receipt=prepared_dir/'LIFT_PIPELINE_EXECUTION.json'
        receipt.update(status='finalized' if not changed else 'source_changed',
            changed_sources=changed,journals=[j.pointer() for j in journals],
            pipeline_execution_receipt=(dict(path=str(pipeline_receipt),sha256=sha(pipeline_receipt))
                if pipeline_receipt.exists() else None))
        save(receipt_path,receipt);finalized=True
        if changed:raise ValueError('Stream execution entry source changed')
    class ClosingApp:
        def __init__(self,app):self.app=app
        def close(self,*args,**kwargs):
            # SimulationApp.fast_shutdown may os._exit. All journals and the
            # final wrapper receipt must be on disk before the real close.
            try:finalize()
            except BaseException:
                kwargs=dict(kwargs,exit_code=1)
                raise
            finally:self.app.close(*args,**kwargs)
    def launch():
        launcher=_launch_native_gpu0(pipeline.common)
        return SimpleNamespace(app=ClosingApp(launcher.app))
    lift.MultiObjectLiftRuntime=runtime;pipeline.common._launch_native=launch
    try:
        return pipeline.execute(prepared_dir)
    except BaseException:
        receipt['wrapper_error']=traceback.format_exc()
        raise
    finally:
        try:finalize()
        finally:
            lift.MultiObjectLiftRuntime=original_runtime;pipeline.common._launch_native=original_launch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',type=Path,required=True)
    args=parser.parse_args();result=execute(args.prepared_dir)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
