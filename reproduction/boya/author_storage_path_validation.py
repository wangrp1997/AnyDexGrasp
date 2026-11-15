"""Bounded full-size storage comparison. No simulator imports or integration.

Eight fresh writes: four on /tmp and four on tmpfs. Failed files remain evidence;
each compression runs once. Existing source logs are only read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import author_contact_compressor as W
import author_verified_contact_journal as J

BASE = Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1')
SOURCE = Path('/tmp/anydex_verified_full_control_20261115_v1/eb25429ed0e703833ab1c9d879a337cf2a0c79554d05463b941045c9e55923ba/CONTACTS.raw.jsonl')
EXPECTED = dict(sha256='e62a6e4d93d3280669686a40cabcb4a3d46c796e9a3c768109b6a52485965716', bytes=1895914265, records=3024)
SCHEMA = 'anydex_boya_full_storage_path_validation_v1'


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=W.sha(p), bytes=p.stat().st_size)


def filesystem(path):
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    value = subprocess.check_output(['findmnt', '-J', '-T', str(p), '-o', 'TARGET,FSTYPE'], text=True)
    return json.loads(value)['filesystems'][0]


def freeze(root):
    root = Path(root).resolve()
    if SOURCE.stat().st_size != EXPECTED['bytes'] or W.sha(SOURCE) != EXPECTED['sha256']:
        raise ValueError('Original full-size source changed')
    spools = dict(disk=Path('/tmp/anydex_storage_path_validation_20261115_v1'),
                  tmpfs=Path('/dev/shm/anydex_storage_path_validation_20261115_v1'))
    if filesystem(spools['tmpfs'])['fstype'] != 'tmpfs':
        raise ValueError('Memory spool must actually be tmpfs')
    for name, p in spools.items():
        if p.exists():
            raise FileExistsError('Fresh spool roots only: '+str(p))
        stat = os.statvfs(p.parent)
        if stat.f_bavail * stat.f_frsize < 5 * EXPECTED['bytes']:
            raise ValueError('Insufficient free capacity for four retained full-size files: '+name)
    sources = {str(Path(p).resolve()): W.sha(p) for p in (__file__, J.__file__, W.__file__)}
    p = dict(schema=SCHEMA, source=binding(SOURCE), expected=EXPECTED, sources=sources,
             python=binding(sys.executable), paths={k:str(v) for k,v in spools.items()},
             filesystems={k:filesystem(v) for k,v in spools.items()}, destination_filesystem=filesystem(root.parent),
             groups=['disk', 'tmpfs'], writers=4, attempts=8, retries=0,
             per_worker_timeout_s=600, max_Kit_starts=0, physics_steps=0, new_training_labels=0,
             training_updates=0, exclude_training=True,
             policy='Two independent predetermined storage groups, once each; no repairs or automatic physical followup')
    root.mkdir(parents=True, exist_ok=False)
    archive=root/'source_archive';archive.mkdir()
    for source,digest in sources.items():
        target=archive/Path(source).name;target.write_bytes(Path(source).read_bytes())
        if W.sha(target)!=digest:raise ValueError('Source archive changed')
    W.save(root/'PROTOCOL.json', p)
    return p


def verify(root):
    root = Path(root).resolve()
    p = read(root/'PROTOCOL.json')
    if (p['schema']!=SCHEMA or p['writers']!=4 or p['attempts']!=8 or p['retries']!=0
            or p['groups']!=['disk','tmpfs'] or p['expected']!=EXPECTED
            or p['max_Kit_starts']!=0 or p['physics_steps']!=0 or p['exclude_training'] is not True):
        raise ValueError('Frozen bounded storage protocol changed')
    for source,digest in p['sources'].items():
        if W.sha(source)!=digest or W.sha(root/'source_archive'/Path(source).name)!=digest:
            raise ValueError('Frozen source or archive changed')
    if binding(p['python']['path'])!=p['python']:
        raise ValueError('Frozen interpreter changed')
    if filesystem(p['paths']['tmpfs'])['fstype']!='tmpfs':
        raise ValueError('Memory spool filesystem changed')
    return p


def independent_readback(target, output, p):
    expected=p['expected']
    command=[p['python']['path'], '-I', '-S', '-B', str(Path(W.__file__).resolve()),
             '--inspect-existing', str(target), '--receipt', str(output),
             '--sha256', expected['sha256'], '--bytes', str(expected['bytes']), '--records', str(expected['records'])]
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
    with output.with_suffix('.log').open('xb') as log:
        result=subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    value=read(output)
    if result.returncode!=0 or value['status']!='verified':
        raise ValueError('Independent full readback failed: '+str(value.get('error')))
    return binding(output)


def worker(root, group, index):
    root=Path(root).resolve();p=verify(root);directory=root/group/f'writer_{index:02d}'
    directory.mkdir(parents=True, exist_ok=False);began=time.monotonic();journal=None
    result=dict(status='failed', group=group, index=index, pid=os.getpid(), Kit_starts=0,
                physics_steps=0, new_training_labels=0, exclude_training=True, retries=0)
    try:
        journal=J.ContactJournal(directory/'CONTACTS.jsonl.gz', source_bindings=p['sources'],
                                 spool_root=p['paths'][group], python=p['python']['path'])
        with Path(p['source']['path']).open('rb') as source:
            for line in source:
                journal.append(json.loads(line))
                if journal.records%256==0:
                    W.save(directory/'PROGRESS.json', dict(records=journal.records, elapsed_s=time.monotonic()-began))
        intended=dict(sha256=journal.logical_hash.hexdigest(), bytes=journal.logical_bytes, records=journal.records)
        if intended!=p['expected']:
            raise ValueError('Full original logical bytes changed before sealing')
        pointer=journal.pointer();W.save(directory/'POINTER.json', pointer)
        result.update(pointer=binding(directory/'POINTER.json'),
                      independent_readback=independent_readback(journal.path, directory/'FIRST_READBACK.json', p),
                      status='verified', logical=intended, raw_spool=str(journal.raw_path))
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        if journal is not None and not journal.closed:
            # An incomplete append must not be turned into a passing full log.
            journal.raw.flush();os.fsync(journal.raw.fileno());journal.raw.close()
    result['wall_s']=time.monotonic()-began
    W.save(directory/'RESULT.json', result)
    return result


def run(root):
    root=Path(root).resolve();p=verify(root);marker=root/'RUN_STARTED.json'
    with marker.open('x') as f:
        json.dump(dict(pid=os.getpid(), unix_s=time.time(), protocol=binding(root/'PROTOCOL.json')), f)
        f.flush();os.fsync(f.fileno())
    began=time.monotonic();rows=[];groups=[]
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','LD_PRELOAD')}
    for group in p['groups']:
        started=time.monotonic();processes=[]
        try:
            for index in range(p['writers']):
                log=(root/f'{group}_{index:02d}.log').open('xb')
                command=[p['python']['path'], '-I','-S','-B',str(Path(__file__).resolve()), 'worker',
                         '--root',str(root),'--group',group,'--index',str(index)]
                process=subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                log.close();processes.append((index, process))
            W.save(root/f'{group}_PROCESSES.json', dict(processes=[dict(index=i,pid=x.pid,pgid=x.pid)for i,x in processes]))
            for index,process in processes:
                process.wait(timeout=max(1, p['per_worker_timeout_s']-(time.monotonic()-started)))
        finally:
            for _,process in processes:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        group_rows=[]
        for index,process in processes:
            file=root/group/f'writer_{index:02d}'/'RESULT.json'
            value=read(file) if file.exists() else dict(status='failed',group=group,index=index,error='Worker result missing')
            value.update(returncode=process.returncode, process_exited=True)
            if process.returncode!=0:value['status']='failed'
            rows.append(value);group_rows.append(value)
        groups.append(dict(group=group,wall_s=time.monotonic()-started,all_verified=all(x['status']=='verified'for x in group_rows)))
        W.save(root/'PROGRESS.json',dict(completed_groups=groups,completed_workers=len(rows),elapsed_s=time.monotonic()-began))
    for value in rows:
        if value['status']!='verified':continue
        directory=root/value['group']/f"writer_{value['index']:02d}"
        try:
            pointer=read(directory/'POINTER.json');J.verify_pointer(pointer)
            value['final_readback']=independent_readback(Path(pointer['data']['path']),directory/'FINAL_READBACK.json',p)
            raw=Path(value['raw_spool'])
            if raw.stat().st_size!=p['expected']['bytes'] or W.sha(raw)!=p['expected']['sha256']:
                raise ValueError('Retained raw file changed after compression')
        except BaseException:
            value.update(status='failed',final_readback_error=traceback.format_exc())
    verify(root)
    if binding(p['source']['path'])!=p['source']:raise ValueError('Original source changed')
    result=dict(schema=SCHEMA,status='storage_paths_verified' if len(rows)==8 and all(x['status']=='verified'for x in rows) else 'failed',
                groups=groups,workers=rows,wall_s=time.monotonic()-began,Kit_starts=0,physics_steps=0,
                new_training_labels=0,training_updates=0,retries=0,production_resumed=False,
                limitation='Finite storage-only replay, not simulator integration, root-cause proof, or crash/power-loss durability')
    W.save(root/'RESULTS.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','worker']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--group',choices=['disk','tmpfs']);parser.add_argument('--index',type=int,choices=range(4))
    a=parser.parse_args()
    result=freeze(a.root)if a.action=='freeze'else run(a.root)if a.action=='run'else worker(a.root,a.group,a.index)
    print(json.dumps({k:result.get(k)for k in ('status','wall_s','physics_steps','new_training_labels')}),flush=True)
    if a.action!='freeze'and result['status']not in ('verified','storage_paths_verified'):raise SystemExit(1)
