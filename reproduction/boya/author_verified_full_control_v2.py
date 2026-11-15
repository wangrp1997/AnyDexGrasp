"""One independent four-trial validation using tmpfs raw contact journals.

The frozen v1 controller and numerical auditor remain unchanged. This protocol
adds the actual interpreter binding missing from v1 and changes only raw spool
storage. Never restarts a prior run or resumes production.
"""
import argparse
import copy
from pathlib import Path
import sys

import author_verified_full_control as OLD
import author_storage_path_validation as STORAGE
import author_verified_contact_journal as J

B, C = OLD.B, OLD.C
REFERENCE = OLD.BASE/'verified_full_control_20261115_v1'
SPOOL = '/dev/shm/anydex_verified_full_control_20261115_v2'


def prepare_protocol(root, previous, readiness):
    root=Path(root).resolve();p=copy.deepcopy(previous)
    # Keep the venv entry path: resolving its symlink would drop site packages.
    p['inputs']['python']=dict(B.binding(sys.executable),path=str(Path(sys.executable).absolute()))
    p['inputs']['storage_path_validation']=B.binding(readiness)
    p['execution_storage_revision']='tmpfs_v2'
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT']=SPOOL
    for name in ('author_verified_full_control_v2.py','author_storage_path_validation.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    p['groups'][0]['workers']=[dict(worker_id=f'worker_{i:02d}',prepared_dir=p['prepared_template']['path'],lanes=1,
        stage_dir=str(root/'cpu_4'/f'worker_{i:02d}'),output_dir=str(root/'cpu_4'/f'worker_{i:02d}'/'result'))for i in range(4)]
    p['scope']='Once-only four complete known-template diagnostic executions; tmpfs raw spool; zero training labels or production dispatch'
    return p


def check_readiness(path):
    path=Path(path).resolve();value=B.read(path)
    if (value['status']!='storage_paths_verified' or value['Kit_starts']!=0 or value['physics_steps']!=0
            or len(value['workers'])!=8 or value['production_resumed']is not False):
        raise ValueError('Both four-writer full-size storage groups must pass first')
    protocol=STORAGE.verify(path.parent)
    for w in value['workers']:
        if w['status']!='verified'or w['logical']!=protocol['expected']or w['returncode']!=0:
            raise ValueError('Incomplete full-size storage readiness')
        B.verify({k:w[k]for k in ('pointer','independent_readback','final_readback')})
        pointer=B.read(w['pointer']['path']);J.verify_pointer(pointer)
        for key in ('independent_readback','final_readback'):
            if B.read(w[key]['path'])['logical_readback']!=protocol['expected']:
                raise ValueError('Independent complete readback failed')
    return value


def freeze(root, readiness):
    check_readiness(readiness)
    if STORAGE.filesystem(SPOOL)['fstype']!='tmpfs':raise ValueError('Raw spool is not tmpfs')
    if Path(SPOOL).exists():raise FileExistsError('Fresh spool required')
    stat=__import__('os').statvfs(Path(SPOOL).parent)
    if stat.f_bavail*stat.f_frsize<5*STORAGE.EXPECTED['bytes']:raise ValueError('Insufficient tmpfs capacity')
    previous=OLD.verify(REFERENCE);root=Path(root).resolve()
    p=prepare_protocol(root,previous,readiness)
    root.mkdir(parents=True,exist_ok=False);p['source_archive']=C.archive_sources(root,p['sources'])
    B.save(root/OLD.PROTOCOL,p,exclusive=True)
    return verify(root)


def verify(root):
    p=OLD.verify(root)
    if (p.get('execution_storage_revision')!='tmpfs_v2'or p['env']['ANYDEX_CONTACT_SPOOL_ROOT']!=SPOOL
            or p['inputs']['python']['path']!=str(Path(sys.executable).absolute())
            or STORAGE.filesystem(SPOOL)['fstype']!='tmpfs'):
        raise ValueError('Frozen tmpfs/interpreter validation protocol changed')
    B.verify({'storage_path_validation':p['inputs']['storage_path_validation']})
    return p


def run(root):
    verify(root)
    result=OLD.run(root)
    verify(root)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--readiness',type=Path)
    a=parser.parse_args();result=freeze(a.root,a.readiness)if a.action=='freeze'else run(a.root)
    print({k:result.get(k)for k in ('status','wall_s','new_training_labels')},flush=True)
    if a.action=='run'and result['status']!='full_integrity_pass':raise SystemExit(1)
