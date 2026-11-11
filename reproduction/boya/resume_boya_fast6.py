"""Explicit zero-forward environment recovery; frozen experiment stays intact.

Restores the LD_LIBRARY_PATH used by the completed prior external12 CGR run.
Retains the failed batch, progress and log. Reuses all settled scenes and RGBD.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import run_boya_fast6 as driver

ROOT, B = driver.ROOT, driver.B
read, save, sha = driver.read, driver.save, driver.sha
OLD_BATCH = driver.OLD / 'representation_stage/BATCH_REPRESENTATION.json'


def recover(root):
    protocol = read(root / 'protocol.json')
    driver.verify(protocol)
    if sha(root/'protocol.json') != read(root/'EVALUATION_FREEZE.json')['protocol_sha256']:
        raise ValueError('Original frozen protocol changed')
    batch_path = root / 'representation_stage/BATCH_REPRESENTATION.json'
    failed = read(batch_path)
    if (failed.get('status') != 'blocked_or_failed'
            or failed.get('model_load_count') != 0 or failed.get('network_forward_count') != 0
            or failed.get('checkpoint_file_load_count') != 0
            or failed.get('error') != 'libgfortran.so.5: cannot open shared object file: No such file or directory'
            or any(r['status'] != 'not_started' for r in failed['objects'])):
        raise ValueError('Recovery allowed only for the recorded zero-load/zero-forward loader failure')
    for row in failed['objects']:
        dest = Path(row['output_dir'])
        if dest.exists() and any(dest.iterdir()):
            raise ValueError('Forward output exists; refuse repeat')
    amendment = root / 'REPRESENTATION_ENV_RECOVERY.json'
    if amendment.exists():
        raise ValueError('Recovery already attempted; never repeat')
    old = read(OLD_BATCH)
    ld = old['native_dependencies']['LD_LIBRARY_PATH']
    for path in ld.split(os.pathsep):
        if not Path(path).is_dir():
            raise ValueError('Previously validated library directory missing: ' + path)
    status_path = root/'EVALUATION_PROGRESS.json'
    before = read(status_path)
    if [x['stage'] for x in before['stages'] if x['returncode']==0] != ['initialize','observe']:
        raise ValueError('Unexpected preceding stages')
    if [x['stage'] for x in before['stages'] if x['returncode']!=0] != ['represent']:
        raise ValueError('Unexpected failure history')
    snapshot = root/'EVALUATION_PROGRESS_before_env_recovery.json'
    shutil.copyfile(status_path, snapshot)
    params = driver.arguments(root, 'represent')
    params[params.index('--run-dir')+1] = root/'representation_stage_env_v2'
    argv = [sys.executable, '-B', str(B/params[0]), *map(str, params[1:])]
    inputs = {str(p):sha(p) for p in [root/'protocol.json',batch_path,OLD_BATCH,snapshot,root/'objects.json']}
    for row in failed['objects']:
        obs = Path(row['observation_json'])
        inputs[str(obs)] = sha(obs)
        for value in read(obs)['files'].values():
            p = Path(value['path'])
            p = p if p.is_absolute() else obs.parent/p
            if sha(p) != value['sha256']:
                raise ValueError('Observed image changed')
            inputs[str(p)] = value['sha256']
    doc = dict(schema='anydex_boya_fast6_zero_forward_environment_recovery_v1',
        original_protocol_sha256=sha(root/'protocol.json'), source_sha256={str(Path(__file__).resolve()):sha(__file__)},
        input_sha256=inputs, LD_LIBRARY_PATH=ld, argv=argv,
        change='Restore original successful native-library search path; new batch receipt directory only',
        RGBD_reused=True, algorithm_model_seed_and_frame_outputs_unchanged=True,
        failed_attempt_model_loads=0, failed_attempt_forwards=0, new_physics_steps=0,
        validation_object_replacement=False, training_writeback=False)
    save(amendment, doc)
    env=dict(os.environ,LD_LIBRARY_PATH=ld,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with (root/'represent_env_v2.log').open('xb') as out:
        proc=subprocess.run(argv,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
    driver.verify(doc)
    driver.verify(protocol)
    receipt=dict(stage='represent',returncode=proc.returncode,argv=argv,wall_s=time.monotonic()-start,
                 environment_recovery_ref=dict(path=str(amendment),sha256=sha(amendment)))
    before['stages'].append(receipt)
    before.update(stage='represent',status='stage_complete' if proc.returncode==0 else 'infrastructure_error')
    save(status_path,before)
    save(root/'REPRESENTATION_ENV_RECOVERY_RESULT.json',receipt)
    if proc.returncode:
        raise RuntimeError('Environment recovery failed; retain outputs and inspect without automatic replay')
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=('recover-representation','score','plan','execute'),required=True)
    a=p.parse_args();root=a.run_dir.resolve()
    if a.stage=='recover-representation':
        recover(root)
    else:
        amendment=read(root/'REPRESENTATION_ENV_RECOVERY.json')
        driver.verify(amendment)
        os.environ['LD_LIBRARY_PATH']=amendment['LD_LIBRARY_PATH']
        driver.stage_run(root,a.stage)
