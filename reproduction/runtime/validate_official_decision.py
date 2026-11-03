"""Run original AnyDex decision weights on original saved CGR records.

This validates the released decision component, not point-cloud CGR generation
or robot/simulation grasp success. No unsafe pickle fallback is permitted.
"""
from __future__ import annotations

import argparse
import ast
import collections.abc
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(path: Path, names: list[str]) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert {node.name for node in nodes} == set(names)
    scope = {'__name__': 'official_decision_overlay', 'torch': torch, 'nn': nn,
             'F': F, 'np': np, 'copy': copy, 'Dataset': Dataset,
             'container_abcs': collections.abc}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def load_original_classes() -> tuple[type, type, type]:
    spec = importlib.util.spec_from_file_location('official_decision_overlay', ROOT/'models/minkowski_graspnet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train_cls = module.MinkowskiGraspNetMultifingerType1
    inference_cls = extract(ROOT/'models/minkowski_graspnet_single_point.py',
                            ['MinkowskiGraspNetMultifingerType1Inference'])['MinkowskiGraspNetMultifingerType1Inference']
    dataset_cls = extract(ROOT/'dataset/multifinger_hand.py', ['MultifingerDataset'])['MultifingerDataset']
    return train_cls, inference_cls, dataset_cls


def safe_state_dict(path: Path, train_cls: type, inference_cls: type) -> tuple[dict, dict]:
    # The published training script torch.save(model), so a narrowly allowlisted
    # weights-only load may be necessary. Never use weights_only=False.
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    known = {
        'torch.nn.modules.conv.Conv1d': nn.Conv1d,
        'torch.nn.modules.batchnorm.BatchNorm1d': nn.BatchNorm1d,
        'torch.nn.modules.dropout.Dropout': nn.Dropout,
        'builtins.set': set,
        '__builtin__.set': set,
        'collections.OrderedDict': collections.OrderedDict,
    }
    for prefix in ['minkowski_graspnet', 'models.minkowski_graspnet',
                   'minkowski_graspnet_single_point', 'models.minkowski_graspnet_single_point',
                   '__main__']:
        known[prefix+'.MinkowskiGraspNetMultifingerType1'] = train_cls
        known[prefix+'.MinkowskiGraspNetMultifingerType1Inference'] = inference_cls
    unknown = [name for name in unsafe if name not in known]
    if unknown:
        raise RuntimeError('Checkpoint contains unapproved pickle globals; not loaded: '+repr(unknown))
    allow = [(known[name], name) for name in unsafe]
    with torch.serialization.safe_globals(allow):
        loaded = torch.load(path, map_location='cpu', weights_only=True)
    container = type(loaded).__name__
    if isinstance(loaded, nn.Module):
        state = loaded.state_dict()
        saved_attributes = {name: getattr(loaded,name) for name in
            ['num_multifinger_type','num_multifinger_depth','num_two_finger_angle','num_two_finger_depth']
            if hasattr(loaded,name)}
    elif isinstance(loaded, dict):
        state = loaded.get('model_state_dict', loaded.get('state_dict', loaded))
        saved_attributes = {}
    else:
        raise TypeError('Unsupported official checkpoint object: '+container)
    if not isinstance(state, dict) or not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError('Checkpoint did not yield a pure tensor state_dict')
    return state, {'unsafe_globals_inspected': unsafe, 'weights_only': True,
                   'allowlist_scope': 'original decision class + Conv1d/BatchNorm1d/Dropout/set only',
                   'container_type': container, 'saved_attributes': saved_attributes,
                   'tensor_count': len(state)}


def confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = scores > threshold
    pos = labels.astype(bool)
    tp = int(np.sum(pos & preds)); fp = int(np.sum(~pos & preds))
    tn = int(np.sum(~pos & ~preds)); fn = int(np.sum(pos & ~preds))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = (2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None
    return {'threshold': threshold, 'comparison': '>', 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy': (tp+tn)/len(labels), 'precision': precision, 'recall': recall, 'f1': f1,
            'selected_predictions': tp+fp}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--hand', choices=['Allegro','Inspire','DH3'])
    parser.add_argument('--finger-type', type=int)
    parser.add_argument('--split', choices=['train','test','unspecified'], default='unspecified')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/official_decision_validation')
    args = parser.parse_args()
    if args.records and (args.hand is None or args.finger_type is None):
        parser.error('--records requires explicit --hand and --finger-type')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    start = time.monotonic()
    report = {
        'scope': 'official pretrained decision component only',
        'not_full_grasp_reproduction': True, 'no_new_training': True,
        'source_checkout': str(ROOT), 'source_revision': 'c9c4a43df33e40860417c7e2dd02f5122d3b2da2',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha(args.checkpoint),
        'python': sys.version, 'python_executable': sys.executable,
        'torch': torch.__version__, 'device': 'cpu',
        'source_sha256': {str(path.relative_to(ROOT)):sha(path) for path in
            [ROOT/'models/minkowski_graspnet.py',ROOT/'models/minkowski_graspnet_single_point.py',
             ROOT/'dataset/multifinger_hand.py',ROOT/'models/loss.py']},
    }
    train_cls, inference_cls, dataset_cls = load_original_classes()
    state, metadata = safe_state_dict(args.checkpoint, train_cls, inference_cls)
    report['checkpoint_load'] = metadata
    report['state_dict_shapes'] = {name:list(value.shape) for name,value in state.items()}
    input_width = int(state['conv1.weight'].shape[1])
    outputs = int(state['conv7.weight'].shape[0])
    report['network_shape'] = {'input_width': input_width, 'output_channels': outputs,
                               'matches_released_robot_inference_head': outputs==20}
    if outputs == 20:
        model = inference_cls(input_num=input_width)
        flavor = 'original_inference_class'
    elif outputs % 20 == 0 and input_width == 480:
        model = train_cls(num_multifinger_type=1, num_multifinger_depth=4,
                          num_two_finger_angle=outputs//20, num_two_finger_depth=5)
        flavor = 'original_training_class_with_checkpoint_matched_dimensions'
    else:
        raise ValueError('Unsupported checkpoint dimensions; refusing to reshape/truncate weights')
    model.load_state_dict(state, strict=True)
    model.eval()
    report['strict_state_dict_load'] = True
    report['model_class'] = flavor
    report['parameter_count'] = sum(param.numel() for param in model.parameters())
    report['loaded_inference_without_inputs'] = True
    if args.records:
        records = json.loads(args.records.read_text())
        if not isinstance(records,dict):
            raise ValueError('Expected original keyed decision JSON mapping')
        prefix = {'Allegro':'Allegro','Inspire':'InspiredHandR','DH3':'DH3'}[args.hand]
        matching = [key for key,info in records.items()
                    if int(info[prefix+'_pose_finger_type'])==args.finger_type]
        source_count = len(matching)
        if args.split == 'train':
            matching = matching[:100]  # exact released MultifingerDataset cap
        if not matching:
            raise ValueError('No records match the requested original hand/finger type')
        dataset = object.__new__(dataset_cls)
        dataset.multifinger_type=args.hand
        dataset.data=matching
        dataset.informations=records
        samples=[dataset.__getitem__(index) for index in range(len(matching))]
        features=np.stack([sample['grasp_preds_features'] for sample in samples])
        if features.shape[1] != input_width:
            raise ValueError('Original features/checkpoint input widths differ')
        if not np.isfinite(features).all():
            raise ValueError('Original feature contains non-finite values')
        labels=np.array([int(sample['result'][0]) for sample in samples])
        two_depths=np.array([int(sample['two_fingers_pose_depth_type'][0]) for sample in samples])
        hand_depths=np.array([int(sample['multifinger_pose_depth_type'][0]) for sample in samples])
        selected_index=two_depths*4+hand_depths  # original MultifingerType1Loss formula
        if not np.all((two_depths>=0)&(two_depths<5)&(hand_depths>=0)&(hand_depths<4)):
            raise ValueError('Record depth outside original decision indexing domain')
        predictions=[]
        inference_start=time.monotonic()
        with torch.no_grad():
            for offset in range(0,len(matching),args.batch_size):
                tensor=torch.from_numpy(features[offset:offset+args.batch_size])
                if outputs==20:
                    pred,_=model(tensor)
                else:
                    count=tensor.shape[0]
                    ep={key:torch.from_numpy(np.stack([sample[key] for sample in samples[offset:offset+count]]))
                        for key in ['two_fingers_pose_angle_type','two_fingers_pose_depth_type',
                                    'multifinger_pose_finger_type','multifinger_pose_depth_type','if_flip']}
                    ep['grasp_preds_features']=tensor
                    pred,_=model(ep)
                predictions.append(pred.reshape(tensor.shape[0],-1).cpu().numpy())
        all_scores=np.concatenate(predictions)
        selected=all_scores[np.arange(len(matching)),selected_index]
        if not np.isfinite(all_scores).all():
            raise ValueError('Checkpoint inference produced non-finite outputs')
        report['real_record_evaluation']={
            'records':str(args.records.resolve()),'records_sha256':sha(args.records),
            'hand':args.hand,'finger_type':args.finger_type,'declared_split':args.split,
            'independent_test_claim':False,
            'split_boundary':'Supplied file/split retained; training membership and object-level leakage not independently established',
            'all_file_records':len(records),'matching_records_before_train_cap':source_count,
            'evaluated_records':len(matching),'labels_positive':int(labels.sum()),
            'labels_negative':int(len(labels)-labels.sum()),
            'score_min':float(selected.min()),'score_max':float(selected.max()),
            'scores_are_sigmoid_probabilities':False,
            'mse_selected_binary_target':float(np.mean((selected-labels)**2)),
            'threshold_metrics':[confusion(labels,selected,threshold) for threshold in [.5,.7,.9]],
            'inference_seconds_cpu':time.monotonic()-inference_start,
            'includes_CGR_generation':False,'includes_physics':False,
        }
        with (args.output_dir/'record_predictions.jsonl').open('w') as stream:
            for index,key in enumerate(matching):
                stream.write(json.dumps({'record_id':key,'label':int(labels[index]),
                    'two_finger_depth_type':int(two_depths[index]),'hand_depth_type':int(hand_depths[index]),
                    'selected_output_index':int(selected_index[index]),'selected_score':float(selected[index]),
                    'all_output_scores':all_scores[index].tolist()})+'\n')
    report['wall_seconds']=time.monotonic()-start
    (args.output_dir/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        print(traceback.format_exc(),file=sys.stderr)
        raise
