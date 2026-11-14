"""Public-code training runtime with the published 480->20 architecture.

Calls the untouched author train/eval/save functions. Data must already be
canonicalized exactly once and split by object. This module neither collects
data nor certifies that a simulator's action labels match the author's task.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

import author_protocol as author

RECIPE = dict(epochs=100, batch_size=128, learning_rate=.0005,
              weight_decay=.0005, poly_power=.99, input_dim=480, output_dim=20,
              output_config_override='num_two_finger_angle=1 matching released models',
              loss='author final-ReLU selected-output MSE',
              model_selection='author save_model validation P/R at .5/.7/.9')


def components():
    scope = dict(torch=torch, nn=nn, F=F, np=np, LambdaLR=LambdaLR)
    model = author.definitions('models/minkowski_graspnet.py',
        {'MinkowskiGraspNetMultifingerType1'}, scope)['MinkowskiGraspNetMultifingerType1']
    inference = author.definitions('models/minkowski_graspnet_single_point.py',
        {'MinkowskiGraspNetMultifingerType1Inference'}, scope)['MinkowskiGraspNetMultifingerType1Inference']
    loss = author.definitions('models/loss.py', {'MultifingerType1Loss'}, scope)['MultifingerType1Loss']
    poly = author.definitions('utils/solvers.py', {'LambdaStepLR','PolyLR'}, scope)['PolyLR']
    return model, inference, loss, poly


def validate_arrays(data):
    required = {'features','labels','target_index','trial_id','mesh_sha256','split'}
    if not required <= data.keys():
        raise ValueError('Bound canonical data columns missing')
    n = len(data['labels'])
    if (data['features'].shape != (n,480) or data['features'].dtype != np.float32
            or not np.isfinite(data['features']).all()):
        raise ValueError('Finite canonical float32 Nx480 required')
    if any(data[k].shape != (n,) or data[k].dtype.hasobject for k in required-{'features'}):
        raise ValueError('Equal-length non-object dataset columns required')
    if (data['labels'].dtype != np.int64 or data['target_index'].dtype != np.int64
            or not np.isin(data['labels'],[0,1]).all()
            or not ((data['target_index']>=0)&(data['target_index']<20)).all()):
        raise ValueError('Binary int64 outcomes and raw_depth*4+extra indices required')
    if len(set(map(str,data['trial_id']))) != n:
        raise ValueError('One label per unique executed trial required')
    if not np.isin(data['split'],['train','internal_check']).all():
        raise ValueError('External rows cannot enter training or model selection')
    fit = np.flatnonzero(data['split']=='train')
    check = np.flatnonzero(data['split']=='internal_check')
    if len(fit)<2 or not len(check):
        raise ValueError('Nonempty held-out internal split and at least two fit rows required')
    if set(data['mesh_sha256'][fit]) & set(data['mesh_sha256'][check]):
        raise ValueError('Source object leakage between fit and internal validation')
    return fit,check


class CanonicalDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices, gesture):
        self.data,self.indices,self.gesture = data,list(map(int,indices)),gesture

    def __len__(self): return len(self.indices)

    def __getitem__(self,index):
        i = self.indices[index]
        target = int(self.data['target_index'][i])
        # The original model does not re-canonicalize these features or use
        # angle/flip to change the forward pass. They are canonical already.
        return dict(grasp_preds_features=torch.from_numpy(self.data['features'][i].copy()),
            result=torch.tensor([int(self.data['labels'][i])],dtype=torch.int32),
            two_fingers_pose_angle_type=torch.tensor([0],dtype=torch.int32),
            two_fingers_pose_depth_type=torch.tensor([target//4],dtype=torch.int32),
            multifinger_pose_depth_type=torch.tensor([target%4],dtype=torch.int32),
            multifinger_pose_finger_type=torch.tensor([self.gesture],dtype=torch.int32),
            if_flip=torch.tensor([0],dtype=torch.int32))


class Writer:
    def __init__(self): self.values = {}
    def add_scalar(self,key,value,epoch): self.values[key] = float(value)


def source_scope(model,criterion,optimizer,scheduler,train_loader,check_loader,
                 gesture,output,save_callback,device):
    class TorchSave:
        def __getattr__(self,name): return getattr(torch,name)
        @staticmethod
        def save(model,path): save_callback(model,Path(path))

    train_writer,check_writer = Writer(),Writer()
    logs = []
    scope = author.definitions('train_multifinger.py',
        {'train_one_epoch','eval_one_epoch','save_model'},
        dict(np=np, time=time, torch=TorchSave(), os=os, multifinger_net=model,
             criterion=criterion, optimizer=optimizer, lr_scheduler=scheduler,
             TRAIN_DATALOADER=train_loader, TEST_DATALOADER=check_loader,
             convert_data_to_gpu=lambda d:{k:v.to(device) for k,v in d.items()},
             TRAIN_WRITER=train_writer, TEST_WRITER=check_writer,
             log_string=logs.append, NUM_MULTIFINGER_TYPE=1,
             GRIPPER_TYPE='Boya', MULTIFINGER_TYPE=gesture,EPOCH_CNT=0,LOG_DIR=str(output)))
    return scope,train_writer,check_writer,logs


def fit_head(data,output,*,gesture,sample_cap,seed,device='cpu',
             development_epochs=None,write_checkpoints=True):
    """Train one cap/type with literal author functions; fresh output only.

    Formal runs use100 epochs. development_epochs is solely for synthetic
    integration tests and is marked in the returned result. Three selected
    checkpoints are retained as safe state_dicts, rather than pickled modules
    for every improvement. Selection events and original filenames are kept.
    """
    import hashlib
    import json

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh training output required')
    if type(gesture) is not int or gesture not in (1,6,7,8):
        raise ValueError('Explicit supported Boya gesture required')
    fit,check = validate_arrays(data)
    selected = author.select_training_records(data['trial_id'][fit],sample_cap=sample_cap)
    fit = fit[:len(selected)]
    if development_epochs is not None and (type(development_epochs) is not int or not 1<=development_epochs<=2):
        raise ValueError('Synthetic development override is1..2 epochs only')
    epochs = development_epochs or RECIPE['epochs']
    torch.manual_seed(seed)
    if str(device).startswith('cuda'): torch.cuda.manual_seed_all(seed)
    fit_order,check_order = fit.tolist(),check.tolist()
    # Source Dataset shuffles once after prefix selection; DataLoader also
    # shuffles train each epoch. Fixed seeds are an explicit repeatability aid.
    rng = random.Random(seed);rng.shuffle(fit_order);rng.shuffle(check_order)
    train_data,check_data = CanonicalDataset(data,fit_order,gesture),CanonicalDataset(data,check_order,gesture)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=128,shuffle=True,num_workers=0)
    check_loader = torch.utils.data.DataLoader(check_data,batch_size=128,shuffle=False,num_workers=0)
    model_cls,inference_cls,loss_cls,poly_cls = components()
    config = dict(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=1,num_two_finger_depth=5)
    model = model_cls(**config).to(device)
    criterion = loss_cls(**config,train_type=gesture).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0005)
    # Even short smoke runs retain the formal100-epoch LR denominator.
    scheduler = poly_cls(optimizer,max_iter=100,power=.99,last_step=-1)
    output.mkdir(parents=True)
    report = dict(schema='anydex_boya_literal_author_training_v1',status='running',
        recipe=RECIPE,gesture_id=gesture,sample_cap=sample_cap,fit_records=len(fit),
        internal_records=len(check),seed=seed,epochs=epochs,
        development_only=development_epochs is not None,
        fit_trial_ids=list(map(str,data['trial_id'][fit])),
        internal_trial_ids=list(map(str,data['trial_id'][check])),
        history=[],selection_events=[],checkpoints={},optimizer_updates=0,
        new_physics_steps=0,CGR_forwards=0,task_semantics_certified=False)
    checkpoint_state = {}

    def save_selected(current,path):
        threshold = path.name.split('_',1)[0]
        state = {k:v.detach().cpu().clone() for k,v in current.state_dict().items()}
        event = dict(epoch=int(scope['EPOCH_CNT']),threshold=threshold,original_filename=path.name)
        report['selection_events'].append(event)
        checkpoint_state[threshold] = state
        if write_checkpoints:
            dst = output/f'best_{threshold}.pt'
            torch.save(dict(schema='anydex_boya_safe_author_head_v1',model_state_dict=state,
                gesture_id=gesture,epoch=event['epoch'],recipe=RECIPE,
                source_sha256=author.PINS,fit_trial_ids=report['fit_trial_ids'],
                internal_trial_ids=report['internal_trial_ids']),dst)
            report['checkpoints'][threshold] = dict(path=str(dst),epoch=event['epoch'],
                sha256=hashlib.sha256(dst.read_bytes()).hexdigest())

    scope,train_writer,check_writer,logs = source_scope(model,criterion,optimizer,scheduler,
        train_loader,check_loader,gesture,output,save_selected,device)
    best = [0.]*9
    started = time.monotonic()
    def save_report():
        (output/'TRAINING_RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save_report()
    try:
        for epoch in range(epochs):
            scope['EPOCH_CNT'] = epoch
            before = max((int(s['step']) for s in optimizer.state.values()),default=0)
            lr = scheduler.get_last_lr()[0]
            scope['train_one_epoch']()
            after = max(int(s['step']) for s in optimizer.state.values())
            report['optimizer_updates'] += after-before
            with torch.no_grad(): indicators = scope['eval_one_epoch']()
            scheduler.step()
            best = scope['save_model'](epoch,indicators,best)
            report['history'].append(dict(epoch=epoch,learning_rate=lr,
                train=dict(train_writer.values),internal=dict(check_writer.values),
                optimizer_updates=after-before,best_indicators=list(map(float,best))))
            save_report()
        strict = {}
        probe = next(iter(check_loader))
        for threshold,state in checkpoint_state.items():
            inference = inference_cls().eval()
            if write_checkpoints:
                loaded = torch.load(report['checkpoints'][threshold]['path'],map_location='cpu',weights_only=True)
                state = loaded['model_state_dict']
            inference.load_state_dict(state,strict=True)
            source_model = model_cls(**config).eval();source_model.load_state_dict(state,strict=True)
            with torch.no_grad():
                a = inference(probe['grasp_preds_features'])[0]
                b = source_model(probe)[0]
            if not torch.equal(a,b): raise ValueError('Selected state differs under published inference')
            strict[threshold] = True
        report.update(status='development_complete' if development_epochs else 'complete',
            best_indicators=best,strict_published_inference_reload=strict,
            selected_checkpoint_count=len(checkpoint_state))
        # Original rules may select no checkpoint, e.g. no predicted positives.
        # Do not silently promote the final epoch in that case.
        report['has_deployable_selected_state'] = bool(checkpoint_state)
    except BaseException as exc:
        report.update(status='training_failed',error=repr(exc))
        raise
    finally:
        report['wall_s'] = time.monotonic()-started
        save_report()
    return report
