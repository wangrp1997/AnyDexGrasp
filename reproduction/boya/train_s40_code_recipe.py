#!/usr/bin/env python3
"""Fixed published-code training recipe, ported to four S40 Boya 480->20 heads.

This is a code-default recipe control, not the complete 12-angle/240-output
original default experiment. Only 32-fit internal heads are trained. The fixed
100th epoch is retained; internal metrics are computed only after training.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path('/home/wangrenpeng/AnyDexGrasp')
BOYA = ROOT / 'reproduction/boya'
sys.path.insert(0, str(BOYA))
import numpy as np
import torch
import train_s40_decision as s40
import train_boya_decision as original

SCHEMA = 'anydex_s40_code_recipe_training_v1'
DATASET_SHA256 = 'c85a284be680707a2b3bcaae7950e1981bd907a9e88870c434ced1c22f6e91bb'
PINNED = {
    ROOT / 'train_multifinger.py': '8f12a094c495ce8bf80fe787e7a583203053aa0e9831288cb8c1d5a7f206fbaf',
    original.SOLVER_SOURCE: 'a9f2b274f438a1a29c1ae17c22a1f553dbe978354ba7143222a653e17a38f713',
    Path(s40.__file__).resolve(): 'a4f3ee4f385c361de19ccf702ded226212deb219feca60240aa6c34f1abf9eae',
    Path(original.__file__).resolve(): 'c95b74d9d0ff2f5602664c9eb61df767fd4b3df22acab3d8327c52d6381671c2',
    original.MODEL_SOURCE: '1463ee5804ec72bafbfc0f469041425092817ff6e6b797923db77120e1167b26',
    original.LOSS_SOURCE: '07c0d5de79f969e565b944635f7ed9924a98f4eb6eff5aae4faa008a75473aa8',
}
CONFIG = dict(epochs=100, batch_size=128, learning_rate=5e-4,
    weight_decay=5e-4, optimizer='Adam', scheduler='original PolyLR',
    poly_power=.99, seed=20261106, input_dim=480, output_dim=20)

def recipe_from_ast(path):
    """Read released defaults without importing its hardware/ME/data side effects."""
    path = Path(path)
    text = path.read_text()
    tree = ast.parse(text)
    expected = {'--max_epoch':100, '--batch_size':128,
                '--learning_rate':5e-4, '--weight_decay':5e-4}
    defaults = {}; evidence = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument' and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value in expected):
            flag = node.args[0].value
            kw = {k.arg:k.value for k in node.keywords}
            if flag in defaults: raise ValueError('Duplicate recipe argument')
            defaults[flag] = ast.literal_eval(kw['default'])
            evidence.append(dict(kind='argparse_default', flag=flag, line=node.lineno,
                                 source=ast.get_source_segment(text, node)))
    if defaults != expected: raise ValueError('Published default recipe changed')
    assignments = {n.targets[0].id:n.value for n in tree.body
                   if isinstance(n, ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0], ast.Name)}
    optimizer = assignments.get('optimizer')
    if (not isinstance(optimizer, ast.Call) or ast.unparse(optimizer.func) != 'optim.Adam'
        or {k.arg:ast.unparse(k.value) for k in optimizer.keywords}
           != {'lr':'BASE_LEARNING_RATE', 'weight_decay':'WEIGHT_DECAY'}):
        raise ValueError('Published Adam parameter binding changed')
    scheduler = assignments.get('lr_scheduler')
    if (not isinstance(scheduler, ast.Call) or ast.unparse(scheduler.func)!='PolyLR'
        or {k.arg:ast.unparse(k.value) for k in scheduler.keywords}
           != {'max_iter':'MAX_EPOCH', 'power':'0.99', 'last_step':'start_epoch - 1'}):
        raise ValueError('Published PolyLR parameter binding changed')
    for kind,node in [('optimizer',optimizer),('scheduler',scheduler)]:
        evidence.append(dict(kind=kind,line=node.lineno,source=ast.get_source_segment(text,node)))
    train = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    loops = [n for n in train.body if isinstance(n,ast.For)]
    if len(loops)!=1: raise ValueError('Published epoch loop changed')
    loop=loops[0]
    calls=[n.value for n in loop.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
    if sum(ast.unparse(n.func)=='lr_scheduler.step' for n in calls)!=1:
        raise ValueError('Scheduler must step once directly per epoch')
    evidence.append(dict(kind='scheduler_epoch_step',line=loop.lineno,
        source='lr_scheduler.step() once directly in for epoch in range(start_epoch, MAX_EPOCH)'))
    if ast.literal_eval(assignments['NUM_TWO_FINGER_ANGLE'])!=12:
        raise ValueError('Original architecture configuration changed')
    return dict(path=str(path.resolve()),sha256=original.sha(path),defaults=defaults,
        optimizer='Adam',scheduler='PolyLR',poly_power=.99,scheduler_step_unit='epoch',
        original_two_finger_angle_count=12,original_output_count=240,
        boya_two_finger_angle_count=1,boya_output_count=20,ast_evidence=evidence)

def frozen_sources():
    for path,digest in PINNED.items():
        if original.sha(path)!=digest: raise ValueError('Frozen source changed: '+str(path))
    paths=list(PINNED)+[Path(__file__).resolve(), BOYA/'render_s40_observations.py']
    return {str(p.resolve()):original.sha(p) for p in paths}

def validate_fixed_dataset(path):
    path=Path(path).resolve()
    if original.sha(path)!=DATASET_SHA256: raise ValueError('Use the frozen original435 dataset only')
    manifest,data,inputs=s40.load_dataset(path)
    if manifest['sample_count']!=435: raise ValueError('Expected435 real labeled trials')
    counts={str(g):{split:int(np.sum(a['split']==split))
              for split in ['train','internal_check']} for g,a in data.items()}
    expected={'1':{'train':83,'internal_check':15},'6':{'train':98,'internal_check':22},
              '7':{'train':98,'internal_check':19},'8':{'train':85,'internal_check':15}}
    if counts!=expected: raise ValueError('Frozen435/364fit/71internal counts changed')
    return manifest,data,inputs,counts

def weighted_metric(heads,key):
    count=sum(r[key]['count'] for r in heads.values())
    return dict(count=count,mse=sum(r[key]['count']*r[key]['mse'] for r in heads.values())/count)

def train(manifest_path,output,device='cuda:0',prepare_only=False):
    output=Path(output).resolve()
    if output.exists(): raise ValueError('New output required; no overwrite or resume')
    if str(device)!='cuda:0': raise ValueError('This frozen run uses GPU0 only')
    sources=frozen_sources();recipe=recipe_from_ast(ROOT/'train_multifinger.py')
    manifest,data,inputs,counts=validate_fixed_dataset(manifest_path)
    config=dict(CONFIG)
    plan=dict(schema=SCHEMA,status='prepared',config=config,published_recipe=recipe,
        dataset_manifest=dict(path=str(Path(manifest_path).resolve()),sha256=DATASET_SHA256),
        input_sha256=inputs,source_sha256=sources,stage_record_counts=counts,
        mesh_groups=manifest['mesh_groups'],device='cuda:0',benchmark_only=False,
        selected_epoch=100,model_selection='fixed final100; no epoch or threshold selection',
        internal_metrics_policy='Only final post-training diagnostics; never optimizer/scheduler/epoch/threshold inputs',
        stages=['internal_check'],all40_refit=False,output_architecture='canonical480/angle1/raw5*extra4=20',
        original_default_architecture='angle12/raw5*extra4=240; deliberately not reproduced here',
        code_recipe_control_not_full_paper_reproduction=True,
        cgr_model_loaded=False,cgr_trained=False,new_CGR_forwards=0,new_physics_steps=0,
        external_validation_evaluated=False,EGAD_objects_accessed=0,
        seed_policy='same original adapter: base20261106 + gesture_id for internal_check heads',
        argv=sys.argv,software=dict(python=sys.version,torch=str(torch.__version__),numpy=str(np.__version__)))
    output.mkdir(parents=True);(output/'internal_check').mkdir()
    original.save(output/'PLAN.json',plan)
    report=dict(plan,status='prepared' if prepare_only else 'training_running',
        plan_sha256=original.sha(output/'PLAN.json'),internal_heads={},heads={})
    original.save(output/'TRAINING_RESULTS.json',report)
    if prepare_only: return report
    started=time.monotonic()
    try:
        device=torch.device(device)
        if not torch.cuda.is_available(): raise ValueError('CUDA0 unavailable')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        report['runtime']=dict(gpu_name=torch.cuda.get_device_name(device),
            gpu_capability=list(torch.cuda.get_device_capability(device)),torch_threads=4,
            cudnn_benchmark=False,cudnn_deterministic=True)
        components=original.official_components()
        provenance=dict(dataset_manifest_sha256=DATASET_SHA256,source_sha256=sources)
        for gesture in original.GESTURES:
            a=data[gesture];fit=np.flatnonzero(a['split']=='train');check=np.flatnonzero(a['split']=='internal_check')
            result=original.train_head(gesture,'internal_check',a,fit,check,
                output/'internal_check',device,config,components,provenance,False)
            if result['epoch']!=100 or result['optimizer_steps']!=100:
                raise ValueError('Expected exactly100 final-epoch updates per head')
            report['internal_heads'][str(gesture)]=result
            report['wall_s']=time.monotonic()-started;original.save(output/'TRAINING_RESULTS.json',report)
        report.update(status='complete',new_optimizer_steps=400,
            fit_weighted_metric=weighted_metric(report['internal_heads'],'fit_metric'),
            internal_weighted_metric=weighted_metric(report['internal_heads'],'internal_check_metric'),
            wall_s=time.monotonic()-started)
    except BaseException:
        report.update(status='training_infrastructure_error',error=traceback.format_exc(),wall_s=time.monotonic()-started)
        raise
    finally:
        report['inputs_and_sources_unchanged']=all(Path(p).is_file() and original.sha(p)==h
            for p,h in {**inputs,**sources}.items())
        report['plan_unchanged']=original.sha(output/'PLAN.json')==report['plan_sha256']
        if not report['inputs_and_sources_unchanged'] or not report['plan_unchanged']:
            report['status']='frozen_input_changed'
        original.save(output/'TRAINING_RESULTS.json',report)
    print(json.dumps(dict(status=report['status'],new_optimizer_steps=report['new_optimizer_steps'],
        fit_weighted_metric=report['fit_weighted_metric'],internal_weighted_metric=report['internal_weighted_metric'],
        internal_heads={g:{k:r[k] for k in ['path','sha256']} for g,r in report['internal_heads'].items()},
        wall_s=report['wall_s']),indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cuda:0'],default='cuda:0')
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();train(a.dataset_manifest,a.output_dir,a.device,a.prepare_only)

if __name__=='__main__':main()
