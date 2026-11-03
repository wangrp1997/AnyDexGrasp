"""Validate every published obj140/480 checkpoint without selecting winners.

Parse/canonicalize each hand's genuine test JSON once. Use the author's hand
remapping receipts and exact original Dataset.__getitem__. No new training,
CGR generation, simulator or robot action occurs.
"""
from __future__ import annotations
import argparse
import collections
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from validate_official_decision import ROOT, sha, load_original_classes, safe_state_dict, confusion


def prepare_groups(path,hand,dataset_cls):
    records=json.loads(path.read_text())
    prefix={'Allegro':'Allegro','Inspire':'InspiredHandR','DH3':'DH3'}[hand]
    groups={}
    for finger in sorted({int(info[prefix+'_pose_finger_type']) for info in records.values()}):
        keys=[key for key,info in records.items() if int(info[prefix+'_pose_finger_type'])==finger]
        dataset=object.__new__(dataset_cls)
        dataset.multifinger_type=hand;dataset.data=keys;dataset.informations=records
        samples=[dataset.__getitem__(index) for index in range(len(keys))]
        features=np.stack([sample['grasp_preds_features'] for sample in samples])
        labels=np.array([int(sample['result'][0]) for sample in samples])
        td=np.array([int(sample['two_fingers_pose_depth_type'][0]) for sample in samples])
        hd=np.array([int(sample['multifinger_pose_depth_type'][0]) for sample in samples])
        if not np.all((td>=0)&(td<5)&(hd>=0)&(hd<4)):
            raise ValueError('Original depth indexing outside domain')
        groups[finger]={'ids':keys,'features':features,'labels':labels,'selected_indices':td*4+hd,
                        'metadata':{key:torch.from_numpy(np.stack([sample[key] for sample in samples]))
                            for key in ['two_fingers_pose_angle_type','two_fingers_pose_depth_type',
                                        'multifinger_pose_finger_type','multifinger_pose_depth_type','if_flip']}}
    metadata={'path':str(path.resolve()),'sha256':sha(path),'records':len(records),
              'types':{str(key):len(value['ids']) for key,value in groups.items()},
              'parse_and_canonicalization_passes':1,'original_dataset_function':'MultifingerDataset.__getitem__'}
    del records,dataset
    gc.collect()
    return groups,metadata


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=Path('/mnt/public/datasets/AnyDexGrasp/metadata/model_obj140_manifest_20261103.json'))
    parser.add_argument('--records-root',type=Path,default=ROOT/'outputs/official_record_preprocessing')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/official_decision_validation_all_obj140')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    start=time.monotonic()
    manifest=json.loads(args.manifest.read_text())
    train_cls,inference_cls,dataset_cls=load_original_classes()
    groups={};record_meta={}
    for hand,short in [('Allegro','allegro'),('DH3','dh3'),('Inspire','inspire')]:
        path=args.records_root/short/'test_single_point.json'
        groups[hand],record_meta[hand]=prepare_groups(path,hand,dataset_cls)
        receipt=path.with_name(path.name+'.preprocessing_receipt.json')
        record_meta[hand]['author_preprocessing_receipt']=json.loads(receipt.read_text())
    rows=[]
    for item in manifest['artifacts']:
        path=Path(item['destination'])
        hand={'allegro_model':'Allegro','dh3_model':'DH3','inspire_model':'Inspire'}[item['hand']]
        finger=int(item['grasp_type'])
        row={'hand':hand,'finger_type':finger,'checkpoint':str(path),'checkpoint_sha256':sha(path),
             'checkpoint_bytes':path.stat().st_size,'published_filename':path.name}
        try:
            state,metadata=safe_state_dict(path,train_cls,inference_cls)
            inputs=int(state['conv1.weight'].shape[1]);outputs=int(state['conv7.weight'].shape[0])
            row['network_shape']={'input_width':inputs,'output_channels':outputs}
            row['checkpoint_load']=metadata
            if outputs==20:
                model=inference_cls(input_num=inputs)
            elif inputs==480 and outputs%20==0:
                model=train_cls(num_multifinger_type=1,num_multifinger_depth=4,num_two_finger_angle=outputs//20,num_two_finger_depth=5)
            else:
                raise ValueError('Refusing unsupported checkpoint dimensions')
            model.load_state_dict(state,strict=True);model.eval()
            row['strict_state_dict_load']=True
            group=groups[hand][finger]
            tensor=torch.from_numpy(group['features'])
            with torch.no_grad():
                pred,_=model(tensor if outputs==20 else dict(group['metadata'],grasp_preds_features=tensor))
            all_scores=pred.reshape(len(group['ids']),-1).numpy()
            if not np.isfinite(all_scores).all():
                raise ValueError('Non-finite official model output')
            selected=all_scores[np.arange(len(group['ids'])),group['selected_indices']]
            metrics=[confusion(group['labels'],selected,t) for t in [.5,.7,.9]]
            threshold,ep,er,ef=map(float,path.stem.split('_')[:4])
            matched=next(metric for metric in metrics if metric['threshold']==threshold)
            diffs={key:abs(matched[key]-expected) if matched[key] is not None else None
                   for key,expected in [('precision',ep),('recall',er),('f1',ef)]}
            row.update({'status':'evaluated','records':len(group['ids']),
                'labels_positive':int(group['labels'].sum()),'labels_negative':int(len(group['ids'])-group['labels'].sum()),
                'selected_score_min':float(selected.min()),'selected_score_max':float(selected.max()),
                'mse_selected_binary_target':float(np.mean((selected-group['labels'])**2)),
                'threshold_metrics':metrics,'filename_declared_threshold':threshold,
                'filename_declared_metrics':{'precision':ep,'recall':er,'f1':ef},
                'filename_metric_absolute_errors':diffs,
                'matches_filename_to_four_decimals':all(value is not None and value<=5.1e-5 for value in diffs.values())})
            tag=hand.lower()+'_type'+str(finger)+'_'+path.stem
            with (args.output_dir/(tag+'.predictions.jsonl')).open('w') as stream:
                for index,key in enumerate(group['ids']):
                    stream.write(json.dumps({'record_id':key,'label':int(group['labels'][index]),
                        'selected_index':int(group['selected_indices'][index]),'selected_score':float(selected[index]),
                        'all_scores':all_scores[index].tolist()})+'\n')
        except Exception as exc:
            row.update({'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        rows.append(row)
        print(json.dumps({key:row.get(key) for key in ['hand','finger_type','published_filename','status','records','matches_filename_to_four_decimals','error']}),flush=True)
    unique_ids={(hand,key) for hand,by_type in groups.items() for group in by_type.values() for key in group['ids']}
    report={'scope':'all published obj140/480 decision checkpoints on original released real test features',
        'source_revision':'c9c4a43df33e40860417c7e2dd02f5122d3b2da2','manifest':str(args.manifest),
        'manifest_sha256':sha(args.manifest),'all_manifest_entries_retained':len(rows)==len(manifest['artifacts']),
        'no_best_model_selection':True,'new_training_steps':0,'physical_grasps_executed':0,
        'includes_pointcloud_CGR_generation':False,'independent_heldout_generalization_claim':False,
        'important_boundary':'Original supplied test records may also have selected these checkpoints. Balanced binary record metrics are not physical grasp success rate. Shared hand-type subsets evaluated by multiple checkpoints are not independent samples.',
        'python':sys.version,'torch':torch.__version__,'device':'cpu',
        'source_sha256':{str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'models/minkowski_graspnet.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'dataset/multifinger_hand.py',ROOT/'update_data.py']},
        'checkpoint_count':len(rows),'evaluated_count':sum(row['status']=='evaluated' for row in rows),
        'filename_metric_match_count':sum(row.get('matches_filename_to_four_decimals',False) for row in rows),
        'unique_hand_type_groups':len({(row['hand'],row['finger_type']) for row in rows}),
        'unique_original_hand_record_ids':len(unique_ids),
        'per_checkpoint_record_evaluations':sum(row.get('records',0) for row in rows),
        'records':record_meta,'models':rows,'wall_seconds':time.monotonic()-start}
    (args.output_dir/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['checkpoint_count','evaluated_count','filename_metric_match_count','unique_hand_type_groups','unique_original_hand_record_ids','per_checkpoint_record_evaluations','wall_seconds']},indent=2),flush=True)
    if any(row['status']!='evaluated' for row in rows):
        raise RuntimeError('Some official models could not be evaluated; all failures preserved')


if __name__=='__main__':
    main()
