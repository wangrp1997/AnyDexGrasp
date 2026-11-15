"""Isolated standard-library compressor: validate raw JSONL, then verify readback.

This worker imports no simulator, tensor, project, or model modules. It never
repairs input or retries compression. A failed output remains evidence and does
not receive a passing seal.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zlib

SCHEMA='anydex_boya_contact_seal_v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with temporary.open('x')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    with_directory_fsync(path.parent)


def with_directory_fsync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def inspect_gzip(path):
    logical=hashlib.sha256();size=records=0;last=b''
    with gzip.open(path,'rb')as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            logical.update(block);size+=len(block);records+=block.count(b'\n');last=block[-1:]
    if size and last!=b'\n':raise ValueError('Final JSONL newline missing')
    return dict(sha256=logical.hexdigest(),bytes=size,records=records)


def compress(raw,destination,receipt,expected,*,level=1):
    raw,destination,receipt=map(lambda p:Path(p).resolve(),(raw,destination,receipt))
    if receipt.exists()or destination.exists():raise FileExistsError('Fresh compression output and seal required')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+'.pending')
    began=time.monotonic();before=raw.stat();result=dict(schema=SCHEMA,status='failed',
        raw=dict(path=str(raw),**expected),destination=str(destination),worker_pid=os.getpid(),
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),python=dict(path=sys.executable,sha256=sha(sys.executable)),
        zlib=dict(compiled=zlib.ZLIB_VERSION,runtime=zlib.ZLIB_RUNTIME_VERSION,origin=zlib.__spec__.origin),
        isolated=bool(sys.flags.isolated),no_site=bool(sys.flags.no_site),
        loaded_native_framework_modules=sorted(n for n in sys.modules if any(n==p or n.startswith(p+'.')for p in('torch','numpy','isaaclab','isaacsim','omni','canondex'))),
        removed_environment_names=[k for k in('PYTHONPATH','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONHOME')if k not in os.environ],
        compression_level=level,Kit_starts=0,physics_steps=0,model_loads=0)
    try:
        if level!=1:raise ValueError('Original gzip level1 required')
        logical=hashlib.sha256();size=records=0
        with raw.open('rb')as source,pending.open('xb')as output:
            with gzip.GzipFile(filename='',fileobj=output,mode='wb',compresslevel=level,mtime=0)as archive:
                for line in source:
                    if not line.endswith(b'\n'):raise ValueError('Incomplete raw JSONL record')
                    row=json.loads(line,parse_constant=lambda v:(_ for _ in()).throw(ValueError(v)))
                    records+=1
                    if not isinstance(row,dict)or type(row.get('step'))is not int or row['step']!=records:
                        raise ValueError('Ordered complete raw record sequence required')
                    logical.update(line);size+=len(line)
                    if archive.write(line)!=len(line):raise OSError('Short compressed write')
            output.flush();os.fsync(output.fileno())
        actual=dict(sha256=logical.hexdigest(),bytes=size,records=records)
        if actual!=expected:raise ValueError('Raw readback differs from the before-write logical digest/count')
        after=raw.stat()
        if(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('Raw spool changed during compression')
        if destination.exists():raise FileExistsError('Destination appeared before publication')
        os.replace(pending,destination);with_directory_fsync(destination.parent)
        decoded=inspect_gzip(destination)
        if decoded!=expected:raise ValueError('Published gzip readback differs from original logical bytes')
        result.update(status='verified',logical_readback=decoded,
            compressed=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size),
            file_fsync_completed=True,directory_fsync_completed=True,complete_JSONL_validation=True,
            gzip_CRC_and_readback_verified=True,raw_spool_retained=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error),raw_spool_retained=raw.exists(),
            pending_retained=pending.exists(),destination_retained=destination.exists())
    result['wall_s']=time.monotonic()-began
    save(receipt,result)
    return result


def inspect_stored(path,receipt,expected):
    path,receipt=Path(path).resolve(),Path(receipt).resolve()
    if receipt.exists():raise FileExistsError('Fresh independent readback receipt required')
    began=time.monotonic();before=sha(path)
    value=dict(schema=SCHEMA,status='failed',expected=expected,input=dict(path=str(path),sha256=before,bytes=path.stat().st_size),
        worker_pid=os.getpid(),isolated=bool(sys.flags.isolated),no_site=bool(sys.flags.no_site),
        loaded_native_framework_modules=sorted(n for n in sys.modules if any(n==p or n.startswith(p+'.')for p in('torch','numpy','isaaclab','isaacsim','omni','canondex'))),
        Kit_starts=0,physics_steps=0,model_loads=0)
    try:
        actual=inspect_gzip(path)
        if actual!=expected:raise ValueError('Logical bytes/records differ from frozen reference')
        if sha(path)!=before:raise ValueError('Compressed file changed during inspection')
        value.update(status='verified',logical_readback=actual)
    except BaseException as error:value['error']=repr(error)
    value['wall_s']=time.monotonic()-began;save(receipt,value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in('raw','destination','inspect-existing'):p.add_argument('--'+name,type=Path)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True);p.add_argument('--records',type=int,required=True)
    a=p.parse_args();expected=dict(sha256=a.sha256,bytes=a.bytes,records=a.records)
    if a.inspect_existing:
        if a.raw or a.destination:p.error('Inspection does not write a new gzip')
        result=inspect_stored(a.inspect_existing,a.receipt,expected)
    else:
        if not a.raw or not a.destination:p.error('Raw input and destination required for compression')
        result=compress(a.raw,a.destination,a.receipt,expected)
    print(json.dumps({k:result.get(k)for k in('status','wall_s','error')}))
    if result['status']!='verified':raise SystemExit(1)


if __name__=='__main__':main()
