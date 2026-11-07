#!/usr/bin/env python3
"""Single-difference Boya control: selected-output MSE before final ReLU.

The source AnyDex Inference class and Loss class remain unchanged. Training
captures BN7 output with a differentiable clone before source in-place ReLU.
Deployment still loads the original Inference class and applies its final ReLU.
This is a declared adaptation control, NOT the author's original training.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import numpy as np
import torch
import train_boya_decision as original
import train_s40_code_recipe as base
SCHEMA='anydex_s40_preactivation_control_v1'
VARIANT='preactivation_selected_mse'
BASE_REPORT=ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json'
BASE_REPORT_SHA='447302d26c1e5cccc10f23ae897105a5d845abe661640cfc4ea5fa6d912ccac0'
CONFIG=dict(base.CONFIG)

class PreactivationCapture:
    """Capture one BN7 forward; clone preserves gradient and original inference."""
    def __init__(self,model):self.model=model;self.handle=None;self.value=None
    def __enter__(self):
        if self.handle is not None:raise RuntimeError('Capture already active')
        self.handle=self.model.bn7.register_forward_hook(self._capture);return self
    def _capture(self,module,args,output):
        if self.value is not None:raise RuntimeError('Previous BN7 output was not consumed')
        if output.ndim!=3 or output.shape[1:]!=(20,1):raise ValueError('Expected original BN7 Bx20x1 output')
        self.value=output.clone() # before original F.relu(..., inplace=True)
    def take(self):
        if self.value is None:raise RuntimeError('No captured BN7 output')
        value=self.value;self.value=None
        return value.transpose(1,2).contiguous()
    def __exit__(self,exc_type,exc,tb):
        if self.handle is not None:self.handle.remove()
        self.handle=None;self.value=None


def preactivation_metric(model,a,indices,device):
    model.eval();scores=[]
    with PreactivationCapture(model)as capture,torch.no_grad():
        for start in range(0,len(indices),128):
            ids=indices[start:start+128];model(torch.from_numpy(a['features'][ids]).to(device))
            p=capture.take()[:,0];target=torch.from_numpy(a['target_index'][ids]).to(device)
            scores.extend(p.gather(1,target[:,None]).flatten().cpu().tolist())
    values=np.asarray(scores,np.float64);labels=a['labels'][indices]
    return dict(count=len(indices),mse=float(np.mean((values-labels)**2)),score_min=float(values.min()),score_max=float(values.max()),negative_selected_count=int((values<0).sum()))


def train_head(gesture,a,fit,check,output,device,config,components,provenance):
    cls,loss_cls,scheduler_cls=components;seed=config['seed']+gesture
    torch.manual_seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    kwargs=dict(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5)
    model=cls(**kwargs).to(device)
    criterion=loss_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5,train_type=gesture).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=scheduler_cls(optimizer,max_iter=config['epochs'],power=config['poly_power'],last_step=-1)
    start=time.monotonic();history=[];steps=0
    with PreactivationCapture(model)as capture:
        for epoch in range(1,config['epochs']+1):
            model.train();batches=original.epoch_batches(fit,config['batch_size'],rng);squared_sum=0.;seen=[];lr=optimizer.param_groups[0]['lr']
            for indices in batches:
                x=torch.from_numpy(a['features'][indices]).to(device);labels=torch.from_numpy(a['labels'][indices]).to(device);target=torch.from_numpy(a['target_index'][indices]).to(device)
                optimizer.zero_grad(set_to_none=True);model(x)
                # Single controlled change: original MSE receives unclipped BN7.
                loss=original.selected_loss(criterion,capture.take(),labels,target,gesture)
                if not torch.isfinite(loss):raise ValueError('Nonfinite original MSE on preactivation')
                loss.backward();optimizer.step();squared_sum+=float(loss.detach().cpu())*len(indices);seen.extend(indices.tolist());steps+=1
            scheduler.step();ids=[str(a['trial_id'][i])for i in seen]
            history.append(dict(epoch=epoch,learning_rate=lr,mse=squared_sum/len(fit),samples=len(seen),distinct_samples=len(set(ids)),batch_sizes=list(map(len,batches)),ordered_trial_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
            print('BOYA_PREACTIVATION_EPOCH '+json.dumps(dict(stage='internal_check',gesture_id=gesture,**history[-1])),flush=True)
    model.eval();fit_metric=original.evaluate(model,a,fit,device);check_metric=original.evaluate(model,a,check,device)
    fit_pre=preactivation_metric(model,a,fit,device);check_pre=preactivation_metric(model,a,check,device)
    path=Path(output)/f'g{gesture:02d}.pt';state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
    checkpoint=dict(schema='anydex_boya_decision_state_dict_v1',model_state_dict=state,gesture_id=gesture,input_dim=480,output_dim=20,epoch=config['epochs'],stage='internal_check',benchmark_only=False,model_class='MinkowskiGraspNetMultifingerType1Inference',architecture_kwargs=kwargs,training_config=config,dataset_manifest_sha256=provenance['dataset_manifest_sha256'],source_sha256=provenance['source_sha256'],fit_mesh_sha256=sorted(set(map(str,a['mesh_sha256'][fit]))),fit_trial_ids=list(map(str,a['trial_id'][fit])),supervision='raw_depth_type*4 + extra_depth_index; one executed BN7 preactivation MSE only',training_variant=VARIANT,author_original_training=False,inference_activation='Unchanged source final ReLU',cgr_trained=False)
    torch.save(checkpoint,path)
    reloaded=cls(**kwargs);loaded=torch.load(path,map_location='cpu',weights_only=True);reloaded.load_state_dict(loaded['model_state_dict'],strict=True);reloaded.eval()
    result=dict(path=str(path),sha256=original.sha(path),gesture_id=gesture,input_dim=480,output_dim=20,epoch=config['epochs'],stage='internal_check',benchmark_only=False,training_variant=VARIANT,fit_records=len(fit),fit_positive=int(a['labels'][fit].sum()),fit_negative=int((a['labels'][fit]==0).sum()),fit_mesh_groups=len(set(a['mesh_sha256'][fit])),optimizer_steps=steps,history=history,fit_metric=fit_metric,internal_check_metric=check_metric,fit_preactivation_metric=fit_pre,internal_preactivation_metric=check_pre,wall_s=time.monotonic()-start,strict_state_dict_reload=True,checkpoint_selection='Fixed final100; no internal model/epoch/threshold selection',training_capture_hooks_removed=not model.bn7._forward_hooks)
    original.save(path.with_suffix('.json'),result);return result


def frozen_sources():
    sources=base.frozen_sources()
    for p in [Path(__file__).resolve(),Path(base.__file__).resolve()]:sources[str(p)]=original.sha(p)
    return sources


def train(manifest_path,output,device='cuda:0',prepare_only=False):
    output=Path(output).resolve()
    if output.exists():raise ValueError('New output required; no overwrite or resume')
    if device not in ('cpu','cuda:0'):raise ValueError('Only cpu or cuda:0 allowed')
    if original.sha(BASE_REPORT)!=BASE_REPORT_SHA:raise ValueError('Frozen base training report changed')
    sources=frozen_sources();recipe=base.recipe_from_ast(ROOT/'train_multifinger.py')
    manifest,data,inputs,counts=base.validate_fixed_dataset(manifest_path)
    inputs=dict(inputs);inputs[str(BASE_REPORT)]=BASE_REPORT_SHA
    config=dict(CONFIG)
    plan=dict(schema=SCHEMA,status='prepared',training_variant=VARIANT,author_original_training=False,
        single_difference='Original selected-output MSE receives differentiable BN7 clone before final ReLU; original class/weights/shapes and deployed ReLU unchanged',
        config=config,published_recipe=recipe,base_training_report_ref=dict(path=str(BASE_REPORT),sha256=BASE_REPORT_SHA),
        dataset_manifest=dict(path=str(Path(manifest_path).resolve()),sha256=base.DATASET_SHA256),input_sha256=inputs,source_sha256=sources,stage_record_counts=counts,mesh_groups=manifest['mesh_groups'],device=device,benchmark_only=False,
        selected_epoch=100,model_selection='Fixed final100; no epoch, seed, threshold or internal metric selection',internal_metrics_policy='Only final post-training diagnostics; never optimizer/scheduler/epoch/threshold inputs',
        stages=['internal_check'],all40_refit=False,output_architecture='canonical480/angle1/raw5*extra4=20',original_default_architecture='angle12/raw5*extra4=240; deliberately not reproduced here',code_recipe_control_not_full_paper_reproduction=True,cgr_model_loaded=False,cgr_trained=False,new_CGR_forwards=0,new_physics_steps=0,external_validation_evaluated=False,EGAD_objects_accessed=0,
        seed_policy='Original adapter base20261106+gesture_id; no seed search',argv=sys.argv,software=dict(python=sys.version,torch=str(torch.__version__),numpy=str(np.__version__)))
    output.mkdir(parents=True);(output/'internal_check').mkdir();original.save(output/'PLAN.json',plan)
    report=dict(plan,status='prepared'if prepare_only else'training_running',plan_sha256=original.sha(output/'PLAN.json'),internal_heads={},heads={})
    original.save(output/'TRAINING_RESULTS.json',report)
    if prepare_only:return report
    started=time.monotonic()
    try:
        task_device=torch.device(device)
        if task_device.type=='cuda'and not torch.cuda.is_available():raise ValueError('CUDA0 unavailable')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        report['runtime']=dict(device=device,torch_threads=4,cudnn_benchmark=False,cudnn_deterministic=True)
        if task_device.type=='cuda':report['runtime'].update(gpu_name=torch.cuda.get_device_name(task_device),gpu_capability=list(torch.cuda.get_device_capability(task_device)))
        components=original.official_components();provenance=dict(dataset_manifest_sha256=base.DATASET_SHA256,source_sha256=sources)
        for gesture in original.GESTURES:
            a=data[gesture];fit=np.flatnonzero(a['split']=='train');check=np.flatnonzero(a['split']=='internal_check')
            result=train_head(gesture,a,fit,check,output/'internal_check',task_device,config,components,provenance)
            if result['epoch']!=100 or result['optimizer_steps']!=100:raise ValueError('Expected100 final-epoch updates per head')
            report['internal_heads'][str(gesture)]=result;report['wall_s']=time.monotonic()-started;original.save(output/'TRAINING_RESULTS.json',report)
        report.update(status='complete',new_optimizer_steps=400,fit_weighted_metric=base.weighted_metric(report['internal_heads'],'fit_metric'),internal_weighted_metric=base.weighted_metric(report['internal_heads'],'internal_check_metric'),wall_s=time.monotonic()-started)
    except BaseException:
        report.update(status='training_infrastructure_error',error=traceback.format_exc(),wall_s=time.monotonic()-started);raise
    finally:
        report['inputs_and_sources_unchanged']=all(Path(p).is_file()and original.sha(p)==h for p,h in {**inputs,**sources}.items());report['plan_unchanged']=original.sha(output/'PLAN.json')==report['plan_sha256']
        if not report['inputs_and_sources_unchanged']or not report['plan_unchanged']:report['status']='frozen_input_changed'
        original.save(output/'TRAINING_RESULTS.json',report)
    print(json.dumps(dict(status=report['status'],training_variant=VARIANT,new_optimizer_steps=report['new_optimizer_steps'],fit_weighted_metric=report['fit_weighted_metric'],internal_weighted_metric=report['internal_weighted_metric'],wall_s=report['wall_s']),indent=2));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();train(a.dataset_manifest,a.output_dir,a.device,a.prepare_only)
if __name__=='__main__':main()
