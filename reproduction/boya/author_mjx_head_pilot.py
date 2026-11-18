"""Limited-data real-label training and strict published inference reload.

This is a 16-fit/4-internal engineering pilot, separate from the frozen author
100/300/1000 sample-cap experiments. It starts no simulator or video recorder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import author_collection_batch as B
import author_protocol as A
import author_training as T


def state_sha(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_data(protocol):
    B.verify(protocol['inputs']);B.verify(protocol['sources'])
    manifest=B.read(protocol['inputs']['manifest']['path'])
    if not manifest['formal_training_eligible'] or manifest['training_updates']!=0:
        raise ValueError('Audited, unmodified source cohort required')
    with np.load(protocol['inputs']['dataset']['path'],allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    fit,check=T.validate_arrays(data)
    if len(fit)!=16 or len(check)!=4 or data['features'].shape!=(20,480):
        raise ValueError('This pilot binds exactly the first 16-fit/4-internal cohort')
    for i,(path,digest) in enumerate(zip(data['record_json'],data['record_sha256'])):
        if B.sha(str(path))!=str(digest):raise ValueError('Formal record changed')
        record=B.read(str(path))
        if not record['eligible'] or not record['formal_training_eligible'] or record['exclude_training']:
            raise ValueError('No promotion of null/diagnostic records')
        if record['label']!=int(data['labels'][i]):raise ValueError('Dataset label changed')
    return data,fit,check


def freeze(root,manifest_path):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    manifest_path=Path(manifest_path).resolve();manifest=B.read(manifest_path)
    sources={n:B.binding(B.HERE/n) for n in (Path(__file__).name,'author_protocol.py','author_training.py')}
    sources.update({n:B.binding(B.ROOT/n) for n in A.PINS})
    p=dict(schema='anydex_mjx_real_label_head_pilot_v1',seed=2026111803,gesture=1,
        limited_data_pilot=True,formal_author_sample_cap_experiment=False,
        source_fit_records=16,source_internal_records=4,epochs=100,recipe=T.RECIPE,
        class_balancing=False,early_stopping=False,seed_retries=0,device='cpu',cpu_threads=1,
        model_input='Source canonical float32 CGR480, no additional canonicalization',
        video_recording=False,new_physics_steps=0,new_CGR_forwards=0,
        automatically_installed_in_collection=False,
        inputs=dict(manifest=B.binding(manifest_path),dataset=manifest['dataset']),sources=sources)
    data,fit,check=load_data(p)
    p['label_counts']={name:dict(records=len(ids),positive=int(data['labels'][ids].sum()),
        negative=int((data['labels'][ids]==0).sum())) for name,ids in [('fit',fit),('internal',check)]}
    archive=root/'source_archive';archive.mkdir()
    for name,source in sources.items():
        destination=archive/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source['path'],destination)
    B.save(root/'PROTOCOL.json',p,exclusive=True)
    return p


def run(root):
    root=Path(root).resolve();p=B.read(root/'PROTOCOL.json');data,fit,check=load_data(p)
    if (root/'RUN_STARTED.json').exists():raise ValueError('No dispatched pilot retry')
    B.save(root/'RUN_STARTED.json',dict(started_unix_s=time.time(),protocol=B.binding(root/'PROTOCOL.json')),exclusive=True)
    torch.set_num_threads(1);torch.manual_seed(p['seed']);np.random.seed(p['seed'])
    order=random.Random(p['seed']);fit_order=fit.tolist();check_order=check.tolist()
    order.shuffle(fit_order);order.shuffle(check_order)
    train_loader=torch.utils.data.DataLoader(T.CanonicalDataset(data,fit_order,1),batch_size=128,shuffle=True,num_workers=0)
    check_loader=torch.utils.data.DataLoader(T.CanonicalDataset(data,check_order,1),batch_size=128,shuffle=False,num_workers=0)
    model_cls,inference_cls,loss_cls,poly_cls=T.components()
    cfg=dict(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5)
    model=model_cls(**cfg).to('cpu');criterion=loss_cls(**cfg,train_type=1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0005)
    scheduler=poly_cls(optimizer,max_iter=100,power=.99,last_step=-1)
    initial_parameter_sha=state_sha(dict(model.named_parameters()))
    selected={};began=time.monotonic()
    report=dict(status='running',limited_data_pilot=True,formal_author_sample_cap_experiment=False,
        protocol=B.binding(root/'PROTOCOL.json'),label_counts=p['label_counts'],epochs=100,
        optimizer_updates=0,history=[],selection_events=[],initial_parameter_sha256=initial_parameter_sha,
        new_physics_steps=0,new_CGR_forwards=0,video_recording=False,
        automatically_installed_in_collection=False)
    def save_checkpoint(current,dest,selection):
        state={k:v.detach().cpu().clone() for k,v in current.state_dict().items()}
        torch.save(dict(schema='anydex_boya_safe_author_head_v1',model_state_dict=state,gesture_id=1,
            epoch=int(scope['EPOCH_CNT']),recipe=T.RECIPE,source_sha256=A.PINS,
            limited_data_pilot=True,selection=selection,protocol=B.binding(root/'PROTOCOL.json'),
            fit_trial_ids=list(map(str,data['trial_id'][fit])),internal_trial_ids=list(map(str,data['trial_id'][check]))),dest)
        return B.binding(dest)
    def save_selected(current,path):
        threshold=path.name.split('_',1)[0]
        dest=root/f'best_{threshold}.pt'
        selected[threshold]=save_checkpoint(current,dest,'original_validation_selection_'+threshold)
        report['selection_events'].append(dict(epoch=int(scope['EPOCH_CNT']),threshold=threshold,original_filename=path.name))
    scope,train_writer,check_writer,logs=T.source_scope(model,criterion,optimizer,scheduler,
        train_loader,check_loader,1,root,save_selected,'cpu')
    best=[0.]*9
    try:
        for epoch in range(100):
            scope['EPOCH_CNT']=epoch
            before=max((int(s['step']) for s in optimizer.state.values()),default=0)
            scope['train_one_epoch']()
            after=max(int(s['step']) for s in optimizer.state.values())
            report['optimizer_updates']+=after-before
            with torch.no_grad():indicators=scope['eval_one_epoch']()
            best=scope['save_model'](epoch,indicators,best);scheduler.step()
            if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                raise ValueError('Nonfinite model state')
            report['history'].append(dict(epoch=epoch,train=dict(train_writer.values),internal=dict(check_writer.values),
                optimizer_updates=after-before,best_indicators=list(map(float,best))))
            B.save(root/'TRAINING_PROGRESS.json',dict(epoch=epoch,epochs=100,optimizer_updates=report['optimizer_updates']))
        if report['optimizer_updates']!=100:raise ValueError('Expected one actual optimizer update per epoch')
        final_parameters=state_sha(dict(model.named_parameters()))
        if final_parameters==initial_parameter_sha:raise ValueError('Model parameters did not change')
        final_checkpoint=save_checkpoint(model,root/'last_pilot.pt','explicit_final_epoch_for_flow_verification')
        reloads={}
        probe=next(iter(check_loader))
        for key,binding in dict(selected,last_pilot=final_checkpoint).items():
            payload=torch.load(binding['path'],map_location='cpu',weights_only=True)
            inference=inference_cls().eval();inference.load_state_dict(payload['model_state_dict'],strict=True)
            source_model=model_cls(**cfg).eval();source_model.load_state_dict(payload['model_state_dict'],strict=True)
            with torch.no_grad():
                a=inference(probe['grasp_preds_features'])[0];b=source_model(probe)[0]
            if not torch.equal(a,b) or not torch.isfinite(a).all():raise ValueError('Published inference reload mismatch')
            reloads[key]=dict(checkpoint=binding,published_inference_exact=True,output_shape=list(a.shape))
        chosen=selected.get('0.5',final_checkpoint)
        payload=torch.load(chosen['path'],map_location='cpu',weights_only=True)
        inference=inference_cls().eval();inference.load_state_dict(payload['model_state_dict'],strict=True)
        with torch.no_grad():all_scores=inference(torch.from_numpy(data['features'].copy()))[0].numpy()
        if all_scores.shape!=(20,1,20) or not np.isfinite(all_scores).all():raise ValueError('Complete finite candidate scores required')
        selected_scores=all_scores[np.arange(20),0,data['target_index']]
        np.savez_compressed(root/'INFERENCE_SCORES.npz',trial_id=data['trial_id'],split=data['split'],labels=data['labels'],
            target_index=data['target_index'],all_depth_scores=all_scores,selected_output_score=selected_scores)
        threshold_metrics={}
        for threshold in (.5,.7,.9):
            pred=selected_scores[check]>=threshold;truth=data['labels'][check].astype(bool)
            threshold_metrics[str(threshold)]=dict(true_positive=int((pred&truth).sum()),false_positive=int((pred&~truth).sum()),
                true_negative=int((~pred&~truth).sum()),false_negative=int((~pred&truth).sum()),
                accuracy=float((pred==truth).mean()))
        report.update(status='limited_data_training_and_inference_complete',optimizer_updates=100,
            final_parameter_sha256=final_parameters,parameters_changed=True,selected_checkpoints=selected,
            final_pilot_checkpoint=final_checkpoint,strict_published_inference_reload=reloads,
            inference_checkpoint=chosen,inference_scores=B.binding(root/'INFERENCE_SCORES.npz'),
            heldout_internal_rows=4,internal_selected_scores=selected_scores[check].tolist(),
            internal_labels=data['labels'][check].tolist(),internal_metrics=threshold_metrics,
            score_semantics='Original ReLU head output; not a calibrated probability',
            new_scene_execution_complete=False,live_candidate_ranking_integrated=False,
            collection_policy_changed=False)
    except BaseException:
        import traceback
        report.update(status='pilot_failed',error=traceback.format_exc())
        raise
    finally:
        report['wall_s']=time.monotonic()-began
        B.save(root/'TRAINING_RESULTS.json',report,exclusive=True)
        B.save(root/'SOURCE_LOG.json',logs,exclusive=True)
        B.verify(p['inputs']);B.verify(p['sources'])
    print({k:report[k] for k in ['status','optimizer_updates','wall_s','label_counts','internal_metrics']},flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':print(freeze(args.root,args.manifest))
    else:run(args.root)
