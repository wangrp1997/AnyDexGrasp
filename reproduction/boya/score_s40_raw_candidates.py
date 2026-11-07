"""S40 internal-only raw score-first diagnostic, with no outcome access/physics.

Reuse locked raw-CGR canonicalization, exact released float32 threshold/global
budget and existing Boya geometry. Only S40 checkpoint/group metadata differ
from the old DGN40/10-specific CLI. The eight diagnostic groups were frozen
before collection; these are not independent external/paper test objects.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import sys,time
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import numpy as np
import torch
from evaluate_boya_decision import Reader,original_inference_class
from deploy_score_first import (make_queries,expand_and_rank,validate_cached_forward,
    PROFILES,OFFICIAL_ROBOT_SHA,admit,save)
from trial_record import array_ref

def evaluate(collection,training_path,output,calibration):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh score-first diagnostic output required')
    reader=Reader()
    for source in [Path(__file__).resolve(),BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'build_collection_pool.py']:
        reader.freeze(source)
    protocol=reader.read(Path(collection)/'protocol.json')
    objects=reader.read(Path(collection)/'objects.json')['objects']
    if protocol['schema']!='anydex_boya_s40_collection_v1':raise ValueError('S40 source protocol required')
    allowed={r['mesh_sha256']for r in protocol['internal_mesh_groups']['fit']}
    heldout={r['mesh_sha256']for r in protocol['internal_mesh_groups']['internal_check']}
    identities={r['object_id']for r in protocol['internal_mesh_groups']['internal_check']}
    if len(allowed)!=32 or len(heldout)!=8 or allowed&heldout or len(identities)!=8:
        raise ValueError('Predeclared disjoint original32/8 required')
    training=reader.read(training_path)
    if (training.get('schema')!='anydex_s40_decision_training_v1' or training['status']!='complete'
        or training['inputs_and_sources_unchanged']is not True or training['external_validation_evaluated']is not False):
        raise ValueError('Complete S40 training with no external evaluation required')
    dataset=training['dataset_manifest'];reader.freeze(dataset['path'],dataset['sha256'])
    model_source=ROOT/'models/minkowski_graspnet_single_point.py';reader.freeze(model_source)
    reader.freeze(ROOT/'robot_dh3.py',OFFICIAL_ROBOT_SHA)
    cls=original_inference_class(model_source,torch);models={};model_metadata={}
    torch.set_num_threads(4)
    for gesture in [1,6,7,8]:
        item=training['internal_heads'][str(gesture)];path=Path(item['path']);reader.freeze(path,item['sha256'])
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        fit=set(checkpoint['fit_mesh_sha256'])
        if (checkpoint['stage']!='internal_check' or checkpoint['gesture_id']!=gesture
            or checkpoint['dataset_manifest_sha256']!=dataset['sha256'] or checkpoint['epoch']!=20
            or not fit or not fit<=allowed or fit&heldout):
            raise ValueError('Internal checkpoint trained on diagnostic or wrong data')
        model=cls(**checkpoint['architecture_kwargs']);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
        models[gesture]=model;model_metadata[str(gesture)]=dict(path=str(path),sha256=item['sha256'],strict_load=True,
            fit_mesh_groups=len(fit),fit_mesh_sha256=sorted(fit),internal_diagnostic_overlap=0)
    profile=copy.deepcopy(PROFILES['released-dh3-v1'])
    output.mkdir(parents=True);summary=[];started=time.monotonic();queries_total=0
    for obj in objects:
        if obj['object_id']not in identities:continue
        out=output/obj['object_id'];out.mkdir()
        obs_path=Path(obj['observation_json']).resolve();obs=reader.read(obs_path)
        frame=Path(obj['representation_dir']).resolve();receipt=reader.read(frame/'RESULTS.json')
        validate_cached_forward(obs_path,obs,frame,receipt,reader);reader.freeze(calibration)
        with np.load(frame/'raw_camera_candidates.npz',allow_pickle=False)as z:
            raw=z['candidates'].copy();features=z['candidate_features'].copy()
        queries,inputs,indices=make_queries(raw,features,profile,'official')
        for gesture,model in models.items():
            ids=[i for i,q in enumerate(queries)if q['gesture_id']==gesture]
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    group=ids[start:start+128];pred,_=model(torch.from_numpy(np.stack([inputs[i]for i in group])))
                    if tuple(pred.shape)!=(len(group),1,20)or not torch.isfinite(pred).all():raise ValueError('Wrong/nonfinite original head output')
                    values=pred[:,0].numpy()
                    for i,row in zip(group,values):queries[i]['all_head_outputs20']=row.tolist()
        queries_total+=len(queries)
        ranked,shortlist=expand_and_rank(queries,profile)
        np.savez_compressed(out/'head_outputs.npz',outputs20=np.asarray([q['all_head_outputs20']for q in queries],np.float32),
            source_indices=np.asarray([q['source_candidate_index']for q in queries]),gesture_ids=np.asarray([q['gesture_id']for q in queries]),
            raw_depth_indices=np.asarray([q['raw_depth_index']for q in queries]),if_flip=np.asarray([q['if_flip']for q in queries]))
        save(out/'shortlist.json',dict(candidates=shortlist,profile='released-dh3-v1',parameters=profile,
            original_source_indices=indices,flip_policy='official',scope='32-fit/8-internal diagnostic; no labels or original ready queue read'))
        reader.verify()
        save(out/'RANKING_FREEZE.json',dict(outputs_sha256={str(p):reader.freeze(p)for p in [out/'head_outputs.npz',out/'shortlist.json']},
            ranking_inputs_sha256=copy.deepcopy(reader.files),internal_models=model_metadata,labels_used_for_ranking=False,
            old_admitted_queue_read=False,dataset_manifest_deserialized=False,geometry_calls_before_freeze=0,arm_calls_before_freeze=0))
        args=argparse.Namespace(stage='geometry',observation=obs_path,frame_dir=frame,calibration=Path(calibration),output_dir=out,max_arm_plans=4)
        geometry=admit(shortlist,raw,obs,args,reader,out/'RANKING_FREEZE.json')if shortlist else dict(geometry_checked=0,geometry_admitted=0,arm_checked=0,selected=None)
        scores=[q['learned_score']for q in ranked]
        result=dict(object_id=obj['object_id'],raw_CGR_topk=len(indices),head_queries=len(queries),expanded_combinations=len(ranked),
            shortlist=len(shortlist),maximum_score=max(scores)if scores else None,score_is_calibrated_probability=False,
            geometry_checked=geometry['geometry_checked'],geometry_admitted=geometry['geometry_admitted'],arm_checked=0,
            selected=None,new_physics_steps=0,new_CGR_forwards=0,threshold_changed=False,old_limited_queue_used=False)
        save(out/'RESULTS.json',result);summary.append(result)
        print('S40_RAW_SCORE_DIAGNOSTIC '+json.dumps(result),flush=True)
    reader.verify()
    report=dict(schema='anydex_s40_raw_score_first_internal_diagnostic_v1',status='complete',objects=summary,
        object_count=len(summary),head_queries=queries_total,models=model_metadata,profile='released-dh3-v1',parameters=profile,
        scope='Unfiltered raw-CGR score-first before geometry, internal8 with32-fit heads; not external test or physical success',
        dataset_manifest_deserialized=False,labels_used_for_ranking=False,old_ready_queue_read=False,
        new_physics_steps=0,new_CGR_forwards=0,new_training_updates=0,wall_s=time.monotonic()-started,
        no_threshold_tuning=True,source_sha256={str(p):reader.freeze(p)for p in [Path(__file__).resolve(),BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py']},
        input_sha256=reader.files)
    save(output/'RESULTS.json',report)
    print(json.dumps(dict(status='complete',objects=len(summary),with_shortlist=sum(x['shortlist']>0 for x in summary),
        with_admitted_geometry=sum(x['geometry_admitted']>0 for x in summary),new_physics_steps=0,wall_s=report['wall_s']),indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-run',type=Path,required=True);p.add_argument('--training-results',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    a=p.parse_args();evaluate(a.collection_run.resolve(),a.training_results.resolve(),a.output_dir,a.calibration.resolve())

if __name__=='__main__':main()
