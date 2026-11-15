"""Full original JSONL bytes, local raw spool, isolated compression and readback.

Only a verified seal can become a pointer consumable by the original auditor.
Raw data and errors are retained on failure; close never retries a compressor.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import author_contact_compressor as W

POINTER_SCHEMA='anydex_boya_contact_journal_pointer_v1'


class JournalIntegrityError(RuntimeError):pass


def verify_pointer(pointer,*,check_data=True):
    if(pointer.get('schema')!=POINTER_SCHEMA or pointer.get('lossless')is not True
        or pointer.get('compression')!='gzip'or pointer.get('record_format')!='jsonl'):
        raise JournalIntegrityError('Original explicit lossless pointer required')
    entry=pointer['integrity_seal'];path=Path(entry['path'])
    if W.sha(path)!=entry['sha256']:raise JournalIntegrityError('Seal changed')
    seal=json.loads(path.read_text())
    expected=pointer['logical']
    if(seal.get('status')!='verified'or seal.get('logical_readback')!=expected or seal.get('compressed')!=pointer['data']
        or seal.get('complete_JSONL_validation')is not True or seal.get('gzip_CRC_and_readback_verified')is not True
        or expected['records']!=pointer['record_count']):raise JournalIntegrityError('Full logical readback seal required')
    if check_data:
        data=pointer['data'];file=Path(data['path'])
        if file.stat().st_size!=data['bytes']or W.sha(file)!=data['sha256']:raise JournalIntegrityError('Published gzip changed')
    return seal


class ContactJournal:
    def __init__(self,path,*,source_bindings=None,spool_root=None,python=None,timeout=300.):
        self.path=Path(path).resolve();self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():raise FileExistsError('Existing contact output is never overwritten')
        self.intent=self.path.with_name(self.path.name+'.INTENT.json')
        self.worker=Path(W.__file__).resolve();self.python=python or sys.executable;self.timeout=float(timeout)
        self.python_binding=dict(path=str(Path(self.python).absolute()),sha256=W.sha(self.python))
        self.source_bindings=dict(source_bindings or {})
        self.source_bindings.update({str(Path(__file__).resolve()):W.sha(__file__),str(self.worker):W.sha(self.worker)})
        root=Path(spool_root or os.environ.get('ANYDEX_CONTACT_SPOOL_ROOT')or Path(tempfile.gettempdir())/'anydex_verified_contacts')
        self.spool=root.resolve()/hashlib.sha256(str(self.path).encode()).hexdigest()
        self.records=self.logical_bytes=0;self.logical_hash=hashlib.sha256()
        self.closed=False;self.seal_error=None;self.seal=None
        self.receipt=self.path.with_name(self.path.name+'.SEAL.json')
        self.state=self.path.with_name(self.path.name+'.STATE.json')
        # An exclusive durable intent prevents a second writer or automatic restart.
        with self.intent.open('x')as stream:
            json.dump(dict(destination=str(self.path),spool=str(self.spool),pid=os.getpid(),unix_s=time.time(),
                source_bindings=self.source_bindings),stream,indent=2);stream.flush();os.fsync(stream.fileno())
        W.with_directory_fsync(self.path.parent)
        self.spool.mkdir(parents=True,exist_ok=False);self.raw_path=self.spool/'CONTACTS.raw.jsonl'
        self.raw=self.raw_path.open('xb')

    def append(self,row):
        if self.closed:raise JournalIntegrityError('Contact journal is finalized')
        if not isinstance(row,dict)or type(row.get('step'))is not int or row['step']!=self.records+1:
            raise ValueError('One ordered complete contact record per native tick required')
        data=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        # Record intended bytes before the write; a short/failed write cannot seal.
        self.logical_hash.update(data);self.logical_bytes+=len(data)
        if self.raw.write(data)!=len(data):raise OSError('Short raw contact write')
        self.records+=1

    def close(self):
        if self.closed:
            if self.seal_error:raise JournalIntegrityError(self.seal_error)
            return
        self.closed=True;began=time.monotonic()
        expected=dict(sha256=self.logical_hash.hexdigest(),bytes=self.logical_bytes,records=self.records)
        value=dict(status='failed',expected=expected,raw_spool=str(self.raw_path),destination=str(self.path),
            records=self.records,source_bindings=self.source_bindings,compressor_attempts=0)
        try:
            try:self.raw.flush();os.fsync(self.raw.fileno())
            finally:self.raw.close()
            W.with_directory_fsync(self.spool)
            if any(W.sha(path)!=digest for path,digest in self.source_bindings.items()):
                raise JournalIntegrityError('Journal/runtime source changed before compression')
            command=[self.python,'-I','-S','-B',str(self.worker),'--raw',str(self.raw_path),
                '--destination',str(self.path),'--receipt',str(self.receipt),'--sha256',expected['sha256'],
                '--bytes',str(expected['bytes']),'--records',str(expected['records'])]
            env={k:v for k,v in os.environ.items()if k not in('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
            value['compressor_attempts']=1
            with(self.spool/'compressor.stdout.log').open('xb')as log:
                process=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=self.timeout)
            if process.returncode!=0:raise JournalIntegrityError('Isolated compressor failed; raw spool and worker receipt retained')
            self.seal=json.loads(self.receipt.read_text())
            if self.seal['raw']!=dict(path=str(self.raw_path),**expected):raise JournalIntegrityError('Worker input expectation changed')
            if(self.seal['source']!=dict(path=str(self.worker),sha256=self.source_bindings[str(self.worker)])
                or self.seal['python']!=self.python_binding or self.seal.get('isolated')is not True
                or self.seal.get('no_site')is not True or self.seal.get('loaded_native_framework_modules')
                or self.seal['compressed']['path']!=str(self.path)):
                raise JournalIntegrityError('Compressor source/interpreter/isolation/output contract changed')
            if any(W.sha(path)!=digest for path,digest in self.source_bindings.items()):
                raise JournalIntegrityError('Journal/runtime source changed during compression')
            pointer=self._pointer();verify_pointer(pointer)
            value.update(status='verified',seal=dict(path=str(self.receipt),sha256=W.sha(self.receipt)),
                compressor_pid=self.seal['worker_pid'],raw_spool_retained=True)
        except BaseException as error:
            self.seal_error=repr(error);value.update(status='failed',error=self.seal_error,raw_spool_retained=self.raw_path.exists())
            raise
        finally:
            value['wall_s']=time.monotonic()-began;W.save(self.state,value)

    def _pointer(self):
        if self.seal is None or self.seal_error:raise JournalIntegrityError('No verified contact seal')
        return dict(schema=POINTER_SCHEMA,compression='gzip',record_format='jsonl',record_count=self.records,
            data=self.seal['compressed'],source_bindings=self.source_bindings,lossless=True,contact_records_omitted=0,
            logical=dict(sha256=self.logical_hash.hexdigest(),bytes=self.logical_bytes,records=self.records),
            integrity_seal=dict(path=str(self.receipt),sha256=W.sha(self.receipt)),
            semantics='Each line is one complete original contact row; logical bytes verified after isolated compression')

    def pointer(self):
        self.close();pointer=self._pointer();verify_pointer(pointer);return pointer
