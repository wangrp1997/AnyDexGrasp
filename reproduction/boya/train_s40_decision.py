"""Train source-bound S40 Boya heads through unchanged original AnyDex components.

Only dataset validation/group metadata change from the old DGN adapter. The
original Inference class, selected-output MSE, optimizer and PolyLR train loop
are reused through a source-frozen read-only adapter module. EGAD is never opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import numpy as np
import torch
import train_boya_decision as original
from render_s40_observations import S40_IDS
GESTURES=original.GESTURES
CONFIG=dict(original.CONFIG)

def sha(p):return original.sha(p)
def read(p):return json.loads(Path(p).read_text())
def save(p,x):return original.save(p,x)

def load_dataset(path):
    path=Path(path).resolve()
    manifest=read(path)
    if (manifest.get('schema')!='anydex_s40_decision_dataset_v1'
        or manifest.get('complete')is not True or manifest.get('collection_complete')is not True
        or manifest.get('benchmark_only')is not False):
        raise ValueError('Complete non-benchmark S40 collected dataset required')
    groups=manifest.get('mesh_groups',{})
    if set(groups)!={'train','internal_check'} or len(groups['train'])!=32 or len(groups['internal_check'])!=8:
        raise ValueError('Predeclared S40 internal32/8 groups required; no external substitute')
    if len(set(groups['train']+groups['internal_check']))!=40:
        raise ValueError('Mesh leakage or duplicate original S40 groups')
    authority={mesh:split for split,meshes in groups.items()for mesh in meshes}
    objects=manifest['objects']
    if len(objects)!=40 or {o['object_id']for o in objects}!={f'graspnet_{i:03d}'for i in S40_IDS} or {o['mesh_sha256']for o in objects}!=set(authority):
        raise ValueError('Dataset objects must cover exactly the original40 mesh groups')
    if any(o.get('outer_split',o.get('source_split','train'))!='train' for o in objects):
        raise ValueError('All S40 source identities are training; no EGAD/test rows')
    byid={o['object_id']:o for o in objects}
    files=manifest['gesture_files']
    if set(files)!=set(map(str,GESTURES)):raise ValueError('Exactly four fixed hand types required')
    inputs={str(path):sha(path)}
    for source,digest in manifest['input_sha256'].items():
        if sha(source)!=digest:raise ValueError('Dataset provenance changed: '+source)
    inputs.update(manifest['input_sha256'])
    data={};seen=set()
    required={'features','labels','target_index','gesture_ids','mesh_sha256','trial_id','object_id','split'}
    for gesture in GESTURES:
        entry=files[str(gesture)];p=Path(entry['path'])
        p=p if p.is_absolute()else path.parent/p
        if sha(p)!=entry['sha256']:raise ValueError('Gesture NPZ changed')
        inputs[str(p.resolve())]=entry['sha256']
        with np.load(p,allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
        if not required<=set(a):raise ValueError('Missing source-bound feature/label columns')
        n=len(a['labels'])
        if a['features'].shape!=(n,480)or a['features'].dtype!=np.float32 or not np.isfinite(a['features']).all():
            raise ValueError('Exact finite float32 canonical480 required')
        for key in required-{'features'}:
            if a[key].shape!=(n,)or a[key].dtype.hasobject:raise ValueError('Unpickled parallel dataset columns required')
        if any(a[k].dtype!=np.int64 for k in ['labels','target_index','gesture_ids']):
            raise ValueError('Labels/depth/finger IDs must be exact integer arrays')
        if (not np.isin(a['labels'],[0,1]).all()or not ((a['target_index']>=0)&(a['target_index']<20)).all()
            or not (a['gesture_ids']==gesture).all()or entry['count']!=n):
            raise ValueError('Invalid real outcome, target or hand type')
        for tid,oid,mesh,split in zip(a['trial_id'],a['object_id'],a['mesh_sha256'],a['split']):
            if str(tid)in seen:raise ValueError('Repeated actual trial identity')
            seen.add(str(tid));obj=byid.get(str(oid))
            if obj is None or obj['mesh_sha256']!=str(mesh) or authority.get(str(mesh))!=str(split):
                raise ValueError('Row crosses original object/internal group identity')
        if sum(a['split']=='train')<2:raise ValueError('At least2 distinct fit records required for original BatchNorm')
        data[gesture]=a
    if len(seen)!=manifest['sample_count']:raise ValueError('Declared sample count differs from unique actual records')
    return manifest,data,inputs

def train(manifest_path,output,device):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh S40 training output required; no silent resume')
    manifest,data,inputs=load_dataset(manifest_path)
    sources={str(p):sha(p)for p in [Path(__file__).resolve(),Path(original.__file__).resolve(),
        original.MODEL_SOURCE,original.LOSS_SOURCE,original.SOLVER_SOURCE,ROOT/'train_multifinger.py',BOYA/'render_s40_observations.py']}
    output.mkdir(parents=True);(output/'heads').mkdir();(output/'internal_check').mkdir()
    config=dict(CONFIG)
    counts={str(g):{split:int(np.sum(a['split']==split))for split in ['train','internal_check']}for g,a in data.items()}
    provenance=dict(dataset_manifest_sha256=sha(manifest_path),source_sha256=sources)
    report=dict(schema='anydex_s40_decision_training_v1',status='training_running',benchmark_only=False,
        dataset_manifest=dict(path=str(Path(manifest_path).resolve()),sha256=sha(manifest_path)),
        input_sha256=inputs,source_sha256=sources,config=config,device=str(device),stage_record_counts=counts,
        heads={},internal_heads={},external_validation_evaluated=False,EGAD_objects_accessed=0,
        split_policy='Original predeclared32 fit/8 internal diagnostic meshes, then refit all40 S40 training groups. No external test.',
        official_components_reused=dict(module=str(Path(original.__file__).resolve()),module_sha256=sha(original.__file__),
            architecture='original MinkowskiGraspNetMultifingerType1Inference480to20',
            loss='original MultfingerType1Loss selected executed output MSE',optimizer='Adam',scheduler='original PolyLR',
            train_loop_function='train_boya_decision.train_head',checkpoint='fixed epoch20; no metric selection',
            configuration_boundary='Paper epochs20/lr1e-4/batch128 with released code WD5e-4/Poly.99; explicit prior adapter config reused'),
        cgr_model_loaded=False,cgr_trained=False,new_physics_steps=0,new_CGR_forwards=0,
        training_is_engineering_port_not_full_real_paper_reproduction=True)
    save(output/'TRAINING_RESULTS.json',report);started=time.monotonic()
    try:
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device=torch.device(device)
        if device.type=='cuda'and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
        components=original.official_components()
        for stage,splits in [('internal_check',['train']),('all_train_meshes',['train','internal_check'])]:
            for gesture,a in data.items():
                fit=np.flatnonzero(np.isin(a['split'],splits))
                check=np.flatnonzero(a['split']=='internal_check')if stage=='internal_check'else np.empty(0,dtype=np.int64)
                result=original.train_head(gesture,stage,a,fit,check,
                    output/('internal_check'if stage=='internal_check'else'heads'),device,config,components,provenance,False)
                report['internal_heads'if stage=='internal_check'else'heads'][str(gesture)]=result
                report['wall_s']=time.monotonic()-started;save(output/'TRAINING_RESULTS.json',report)
        report.update(status='complete',new_optimizer_steps=sum(row['optimizer_steps']for key in ['heads','internal_heads']for row in report[key].values()),
            wall_s=time.monotonic()-started)
    except BaseException:
        report.update(status='training_infrastructure_error',error=traceback.format_exc(),wall_s=time.monotonic()-started)
        raise
    finally:
        unchanged=all(Path(p).is_file()and sha(p)==h for p,h in dict(inputs,**sources).items())
        report['inputs_and_sources_unchanged']=unchanged
        if not unchanged:report['status']='frozen_input_changed'
        save(output/'TRAINING_RESULTS.json',report)
    print(json.dumps(dict(status=report['status'],optimizer_updates=report['new_optimizer_steps'],
        samples=manifest['sample_count'],heads={g:x['path']for g,x in report['heads'].items()},wall_s=report['wall_s']),indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();train(a.dataset_manifest.resolve(),a.output_dir,a.device)

if __name__=='__main__':main()
