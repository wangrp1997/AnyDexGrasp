"""Frozen four-type development dispatch, with distinct once-only stage jobs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
B=ROOT/'reproduction/boya'
PY=sys.executable
ENV=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', CUDA_VISIBLE_DEVICES='0',
    PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='4',
    PYTHONPATH=str(ROOT)+':'+str(B)+':'+str(ROOT/'outputs/native_deps/clarabel_0111_overlay'),
    LD_LIBRARY_PATH=':'.join(map(str,[ROOT/'outputs/native_deps/os_libs',ROOT/'outputs/native_deps/openblas/lib',
        ROOT/'outputs/native_deps/cuda-13.0/lib64',Path('/home/wangrenpeng/tendon-spin/.venv/lib/python3.12/site-packages/torch/lib')])))


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d): Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def job(name, argv):
    directory=OUT/'jobs';directory.mkdir(exist_ok=True)
    path=directory/(name+'.json')
    initial=dict(stage=name,status='running',argv=list(map(str,argv)),
        source_sha256={str(Path(__file__)):sha(__file__)},
        development_protocol_sha256=sha(OUT/'DEVELOPMENT_PROTOCOL.json'),
        environment={k:ENV[k] for k in ('OMNI_KIT_ACCEPT_EULA','CUDA_VISIBLE_DEVICES','PYTHONPATH',
            'LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')})
    with path.open('x') as f: f.write(json.dumps(initial,indent=2)+'\n')
    start=time.monotonic()
    with (directory/(name+'.log')).open('xb') as log:
        proc=subprocess.run(list(map(str,argv)),cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    initial.update(status='finished' if proc.returncode==0 else 'failed',returncode=proc.returncode,
                   wall_s=time.monotonic()-start)
    save(path,initial)
    print(json.dumps({k:initial[k] for k in ('stage','status','returncode','wall_s')}),flush=True)
    return proc.returncode


def main(stage):
    protocol=read(OUT/'DEVELOPMENT_PROTOCOL.json')
    for p,d in protocol['pre_capture_source_sha256'].items():
        assert sha(p)==d,p
    scene=OUT/'scene/capture/observed_scene.json'
    observation=OUT/'scene/capture/observation.json'
    identity='development_three_lift_20261114'
    representation=OUT/'representation/representations'/identity
    if stage=='capture':
        return job(stage,[PY,'-B',B/'author_scene.py','capture','--scene',OUT/'scene/scene.json',
                          '--output-dir',OUT/'scene/capture'])
    if stage=='represent':
        manifest=OUT/'REPRESENTATION_MANIFEST.json'
        with manifest.open('x') as f:
            f.write(json.dumps(dict(schema='anydex_boya_author_scene_representation_manifest_v1',
                scope=protocol['scope'],objects=[dict(object_id=identity,observation_json=str(observation),
                observation_sha256=sha(observation))]),indent=2)+'\n')
        return job(stage,[PY,'-B',B/'run_collection_representation.py','--manifest',manifest,
                         '--run-dir',OUT/'representation','--device','cuda:0','--seed','20261114'])
    if stage=='prepare':
        sources={str(p):sha(p) for p in B.glob('author_*.py')}
        freeze=dict(schema='anydex_boya_lift_development_preparation_queue_v1',
            protocol_sha256=sha(OUT/'DEVELOPMENT_PROTOCOL.json'),source_sha256=sources,
            scene_sha256=sha(scene),observation_sha256=sha(observation),
            representation_receipt_sha256=sha(representation/'RESULTS.json'),
            conditions=protocol['gestures'],physics_steps=0,training_updates=0)
        with (OUT/'PREPARATION_QUEUE.json').open('x') as f: f.write(json.dumps(freeze,indent=2)+'\n')
        def one(row):
            g=row['gesture_id']
            return job('prepare_g'+str(g),[PY,'-B',B/'author_lift_pipeline.py','prepare',
                '--scene',scene,'--representation-dir',representation,'--observation',observation,
                '--gesture-id',g,'--source-type',row['source_type'],'--seed',row['seed'],
                '--calibration',protocol['calibration']['path'],'--cloud-manifest',protocol['cloud_manifest']['path'],
                '--backend',protocol['backend']['path'],'--output-dir',OUT/('g'+str(g))])
        start=time.monotonic()
        with ThreadPoolExecutor(max_workers=3) as pool: codes=list(pool.map(one,protocol['gestures']))
        for p,d in sources.items(): assert sha(p)==d,p
        results=[]
        for row in protocol['gestures']:
            g=row['gesture_id'];path=OUT/('g'+str(g))/'LIFT_PIPELINE_RESULTS.json';result=read(path)
            results.append(dict(gesture_id=g,status=result['status'],result_json=str(path),result_sha256=sha(path),
                                ready_sha256=sha(path.parent/'READY.json') if (path.parent/'READY.json').exists() else None))
        save(OUT/'PREPARATION_RESULTS.json',dict(status='infrastructure_failure' if any(codes) else 'complete',
            wall_s=time.monotonic()-start,conditions=results,source_sha256=sources,
            preparation_queue_sha256=sha(OUT/'PREPARATION_QUEUE.json'),physics_steps=0))
        return 1 if any(codes) else 0
    if stage=='execute':
        prepared=read(OUT/'PREPARATION_RESULTS.json')
        assert prepared['status']=='complete'
        for p,d in prepared['source_sha256'].items(): assert sha(p)==d,p
        with (OUT/'EXECUTION_QUEUE.json').open('x') as f:
            f.write(json.dumps(dict(preparation_results_sha256=sha(OUT/'PREPARATION_RESULTS.json'),
                conditions=prepared['conditions'],max_attempts_per_ready=1,serial_order=[1,6,7,8],
                formal_training_writeback=False),indent=2)+'\n')
        for row in prepared['conditions']:
            assert sha(row['result_json'])==row['result_sha256']
            if row['status']!='ready': continue
            directory=OUT/('g'+str(row['gesture_id']))
            assert sha(directory/'READY.json')==row['ready_sha256']
            code=job('execute_g'+str(row['gesture_id']),[PY,'-B',B/'author_lift_pipeline.py',
                'execute','--prepared-dir',directory])
            if code: return code
        return 0
    raise ValueError(stage)


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))
