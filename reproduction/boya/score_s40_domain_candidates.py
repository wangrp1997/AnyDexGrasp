"""S40 internal score-first diagnostic with explicit Boya pre-head domain.

Uses cached original raw CGR and 32-fit heads, preserving source row/feature
identity through the domain mask and top-K. No labels, old ready queues, CGR
forward, arm planning or physical execution. The Boya policy is declared and
source-bound separately; it is not a literal copy of the DH3 rig rectangle.
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
from boya_prehead_domain import apply_boya_domain
from collections import Counter

def make_domain_queries(raw, features, transform, config, profile, provenance):
    if raw.ndim != 2 or raw.shape[1:] != (15,) or features.shape != (len(raw), 2261):
        raise ValueError('Expected paired original Nx15 / Nx2261 arrays')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite paired CGR features')
    domain = apply_boya_domain(raw, transform, config, source_provenance=provenance)
    kept = domain.keep_indices
    queries, inputs, local = make_queries(raw[kept], features[kept], profile, 'official')
    for query in queries:
        query['source_candidate_index'] = int(kept[query['source_candidate_index']])
    return queries, inputs, [int(kept[i]) for i in local], domain


def evaluate(collection,training_path,output,calibration,domain_policy):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh score-first diagnostic output required')
    reader=Reader()
    for source in [Path(__file__).resolve(),BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'build_collection_pool.py',BOYA/'boya_prehead_domain.py']:
        reader.freeze(source)
    policy = reader.read(domain_policy)
    if policy.get('schema') != 'anydex_boya_prehead_domain_policy_v1' or not policy.get('source_sha256'):
        raise ValueError('Explicit source-bound Boya domain policy required')
    for path, digest in policy['source_sha256'].items():
        reader.freeze(path, digest)
    policy_sha = reader.files[str(Path(domain_policy).resolve())]
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
        queries,inputs,indices,domain=make_domain_queries(raw,features,obs['T_WC'],policy['config'],profile,
            dict(domain_policy=dict(path=str(Path(domain_policy).resolve()),sha256=policy_sha),
                 observation=dict(path=str(obs_path),sha256=reader.files[str(obs_path)]),
                 raw_npz=dict(path=str(frame/'raw_camera_candidates.npz'),sha256=reader.files[str(frame/'raw_camera_candidates.npz')]),
                 policy_sources=policy['source_sha256']))
        save(out/'DOMAIN.json',domain.receipt)
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
        np.savez_compressed(out/'head_outputs.npz',outputs20=np.asarray([q['all_head_outputs20']for q in queries],np.float32).reshape(-1,20),
            source_indices=np.asarray([q['source_candidate_index']for q in queries]),gesture_ids=np.asarray([q['gesture_id']for q in queries]),
            raw_depth_indices=np.asarray([q['raw_depth_index']for q in queries]),if_flip=np.asarray([q['if_flip']for q in queries]))
        save(out/'shortlist.json',dict(candidates=shortlist,profile='released-dh3-v1',parameters=profile,
            original_source_indices=indices,flip_policy='official',scope='32-fit/8-internal diagnostic; no labels or original ready queue read'))
        reader.verify()
        save(out/'RANKING_FREEZE.json',dict(outputs_sha256={str(p):reader.freeze(p)for p in [out/'DOMAIN.json',out/'head_outputs.npz',out/'shortlist.json']},
            ranking_inputs_sha256=copy.deepcopy(reader.files),internal_models=model_metadata,labels_used_for_ranking=False,
            old_admitted_queue_read=False,dataset_manifest_deserialized=False,geometry_calls_before_freeze=0,arm_calls_before_freeze=0))
        args=argparse.Namespace(stage='geometry',observation=obs_path,frame_dir=frame,calibration=Path(calibration),output_dir=out,max_arm_plans=4)
        geometry=admit(shortlist,raw,obs,args,reader,out/'RANKING_FREEZE.json')if shortlist else dict(geometry_checked=0,geometry_admitted=0,arm_checked=0,selected=None)
        geometry_records = json.loads((out/'geometry.json').read_text()) if shortlist else dict(checked=[],admitted=[])
        rejections = Counter(r['geometry']['reason'] for r in geometry_records['checked'] if not r['geometry']['ok'])
        scores=[q['learned_score']for q in ranked]
        result=dict(object_id=obj['object_id'],domain_counts=domain.receipt['counts'],raw_CGR_topk=len(indices),head_queries=len(queries),expanded_combinations=len(ranked),
            shortlist=len(shortlist),maximum_score=max(scores)if scores else None,score_is_calibrated_probability=False,
            geometry_checked=geometry['geometry_checked'],geometry_admitted=geometry['geometry_admitted'],arm_checked=0,
            selected=None,geometry_rejection_counts=dict(rejections),
            first_admitted=geometry_records['admitted'][0] if geometry_records['admitted'] else None,
            new_physics_steps=0,new_CGR_forwards=0,threshold_changed=False,old_limited_queue_used=False)
        save(out/'RESULTS.json',result);summary.append(result)
        print('S40_DOMAIN_SCORE_DIAGNOSTIC '+json.dumps(result),flush=True)
    if len(summary) != 8:raise ValueError('Exactly eight predeclared internal frames required')
    reader.verify()
    report=dict(schema='anydex_s40_domain_score_first_internal_diagnostic_v1',domain_policy=dict(path=str(Path(domain_policy).resolve()),sha256=policy_sha,policy=policy),status='complete',objects=summary,
        object_count=len(summary),head_queries=queries_total,models=model_metadata,profile='released-dh3-v1',parameters=profile,
        scope='Explicit Boya pre-head domain then raw-CGR score-first before geometry, internal8 with32-fit heads; not external test or physical success',
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
    p.add_argument('--domain-policy',type=Path,required=True)
    a=p.parse_args();evaluate(a.collection_run.resolve(),a.training_results.resolve(),a.output_dir,a.calibration.resolve(),a.domain_policy.resolve())

if __name__=='__main__':main()
