"""S40 Boya hand-domain port: calibrated opening and source table support.

The exact table subpredicate runs before scoring; full cloud collisions and
arm admission stay after the immutable global ranking. Four independent CPU
workers check ranked objects. Models, data, raw budget and threshold are fixed.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import sys,time,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import numpy as np
import torch
from evaluate_boya_decision import Reader,original_inference_class
from deploy_score_first import (make_queries,expand_and_rank,validate_cached_forward,
    PROFILES,OFFICIAL_ROBOT_SHA,admit,save,calibrated_slots)
from boya_hand_domain import make_hand_queries,expand_hand_queries,SourceHandSupport,validate_policy
from asset_adapter import BoyaAssets
from collections import Counter


def write_progress(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(path)


def geometry_worker(path):
    reader=Reader();job=reader.read(path)
    for source,digest in job['rank_inputs'].items():reader.freeze(source,digest)
    out=Path(job['output_dir']);obs_path=Path(job['observation']);obs=reader.read(obs_path)
    document=reader.read(out/'shortlist.json');shortlist=document['candidates']
    with np.load(Path(job['frame_dir'])/'raw_camera_candidates.npz',allow_pickle=False)as z:raw=z['candidates'].copy()
    args=argparse.Namespace(stage='geometry',observation=obs_path,frame_dir=Path(job['frame_dir']),
        calibration=Path(job['calibration']),output_dir=out,max_arm_plans=4)
    geometry=admit(shortlist,raw,obs,args,reader,Path(job['ranking_freeze']))if shortlist else dict(geometry_checked=0,geometry_admitted=0,arm_checked=0,selected=None)
    detail=json.loads((out/'geometry.json').read_text())if shortlist else dict(checked=[],admitted=[])
    for row in detail['checked']:
        if row.get('source_slot')!=row['domain_source_slot']:raise ValueError('Post-score mapping changed the pre-head calibrated slot')
    rejections=Counter(row['geometry']['reason']for row in detail['checked']if not row['geometry']['ok'])
    result=dict(job['base_result'],geometry_checked=geometry['geometry_checked'],geometry_admitted=geometry['geometry_admitted'],
        geometry_rejection_counts=dict(rejections),first_admitted=detail['admitted'][0]if detail['admitted']else None,
        support_plane_recheck_rejections=rejections.get('table',0))
    reader.verify();save(out/'RESULTS.json',result)


def evaluate(collection,training_path,output,calibration,domain_policy,head_recipe,geometry_workers=4):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh score-first diagnostic output required')
    reader=Reader()
    for source in [Path(__file__).resolve(),BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'build_collection_pool.py',BOYA/'boya_prehead_domain.py',BOYA/'boya_hand_domain.py']:
        reader.freeze(source)
    policy = reader.read(domain_policy)
    validate_policy(policy)
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
    expected_schema = {'paper20':'anydex_s40_decision_training_v1',
                       'code100':'anydex_s40_code_recipe_training_v1'}[head_recipe]
    expected_epochs = {'paper20':20,'code100':100}[head_recipe]
    expected_lr = {'paper20':1e-4,'code100':5e-4}[head_recipe]
    cfg=training['config']
    expected_config=dict(epochs=expected_epochs,batch_size=128,learning_rate=expected_lr,
        weight_decay=5e-4,optimizer='Adam',scheduler='original PolyLR',poly_power=.99,
        seed=20261106,input_dim=480,output_dim=20)
    if cfg != expected_config:raise ValueError('Head recipe must match the explicitly selected frozen configuration')
    if (training.get('schema')!=expected_schema or training['status']!='complete'
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
            or checkpoint['dataset_manifest_sha256']!=dataset['sha256'] or checkpoint['epoch']!=expected_epochs or checkpoint['training_config']!=expected_config
            or not fit or not fit<=allowed or fit&heldout):
            raise ValueError('Internal checkpoint trained on diagnostic or wrong data')
        model=cls(**checkpoint['architecture_kwargs']);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
        models[gesture]=model;model_metadata[str(gesture)]=dict(path=str(path),sha256=item['sha256'],strict_load=True,
            fit_mesh_groups=len(fit),fit_mesh_sha256=sorted(fit),internal_diagnostic_overlap=0)
    profile=copy.deepcopy(PROFILES['released-dh3-v1'])
    output.mkdir(parents=True);summary=[];pending=[];started=time.monotonic();queries_total=0
    assets=BoyaAssets();families=calibrated_slots(assets,reader.read(calibration),reader);support=SourceHandSupport(assets)
    save(output/'PLAN.json',dict(schema='anydex_s40_hand_domain_plan_v1',policy_sha256=policy_sha,
        training_results=str(Path(training_path).resolve()),head_recipe=head_recipe,geometry_workers=geometry_workers,
        parameters=profile,objects=sorted(identities),labels_read=False,new_physics_steps=0,
        support_predicate='Exact source table prefix before heads; full point-cloud collision after ranking'))
    for obj in objects:
        if obj['object_id']not in identities:continue
        out=output/obj['object_id'];out.mkdir()
        obs_path=Path(obj['observation_json']).resolve();obs=reader.read(obs_path)
        frame=Path(obj['representation_dir']).resolve();receipt=reader.read(frame/'RESULTS.json')
        validate_cached_forward(obs_path,obs,frame,receipt,reader);reader.freeze(calibration)
        with np.load(frame/'raw_camera_candidates.npz',allow_pickle=False)as z:
            raw=z['candidates'].copy();features=z['candidate_features'].copy()
        queries,inputs,indices,domain=make_hand_queries(raw,features,obs['T_WC'],policy,profile,families,assets,
            dict(domain_policy=dict(path=str(Path(domain_policy).resolve()),sha256=policy_sha),
                 observation=dict(path=str(obs_path),sha256=reader.files[str(obs_path)]),
                 raw_npz=dict(path=str(frame/'raw_camera_candidates.npz'),sha256=reader.files[str(frame/'raw_camera_candidates.npz')]),
                 policy_sources=policy['source_sha256']),support=support)
        save(out/'DOMAIN.json',domain)
        for gesture,model in models.items():
            ids=[i for i,q in enumerate(queries)if q['gesture_id']==gesture]
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    group=ids[start:start+128];pred,_=model(torch.from_numpy(np.stack([inputs[i]for i in group])))
                    if tuple(pred.shape)!=(len(group),1,20)or not torch.isfinite(pred).all():raise ValueError('Wrong/nonfinite original head output')
                    values=pred[:,0].numpy()
                    for i,row in zip(group,values):queries[i]['all_head_outputs20']=row.tolist()
        queries_total+=len(queries)
        ranked,shortlist=expand_hand_queries(queries,profile)
        np.savez_compressed(out/'head_outputs.npz',outputs20=np.asarray([q['all_head_outputs20']for q in queries],np.float32).reshape(-1,20),
            source_indices=np.asarray([q['source_candidate_index']for q in queries]),gesture_ids=np.asarray([q['gesture_id']for q in queries]),
            raw_depth_indices=np.asarray([q['raw_depth_index']for q in queries]),if_flip=np.asarray([q['if_flip']for q in queries]))
        save(out/'shortlist.json',dict(candidates=shortlist,profile='released-dh3-v1',parameters=profile,
            original_source_indices=indices,flip_policy='both',scope='32-fit/8-internal diagnostic; no labels or original ready queue read'))
        reader.verify()
        save(out/'RANKING_FREEZE.json',dict(outputs_sha256={str(p):reader.freeze(p)for p in [out/'DOMAIN.json',out/'head_outputs.npz',out/'shortlist.json']},
            ranking_inputs_sha256=copy.deepcopy(reader.files),internal_models=model_metadata,labels_used_for_ranking=False,
            old_admitted_queue_read=False,dataset_manifest_deserialized=False,geometry_calls_before_freeze=0,arm_calls_before_freeze=0,
            geometry_calls_definition='Full visible-point-cloud collision calls; exact source table prefix counted separately',
            support_plane_queries_before_freeze=domain['counts']['support_plane_tests'],
            support_plane_is_full_collision_admission=False,
            query_count_definitions=dict(geometry_calls_before_freeze='full_cloud_collision_calls',
                support_plane_queries_before_freeze='open_hand_table_support_function_queries_not_full_collision_admission'),
            support_plane_predicate='Original .0005 table predicate, including all fingers and pregrasp endpoint; no relaxed gate'))
        scores=[q['learned_score']for q in ranked]
        base=dict(object_id=obj['object_id'],domain_counts=domain['counts'],raw_CGR_topk=len(indices),
            head_queries=len(queries),expanded_combinations=len(ranked),shortlist=len(shortlist),
            maximum_score=max(scores)if scores else None,score_is_calibrated_probability=False,
            arm_checked=0,selected=None,new_physics_steps=0,new_CGR_forwards=0,
            threshold_changed=False,old_limited_queue_used=False)
        job=dict(output_dir=str(out),frame_dir=str(frame),observation=str(obs_path),calibration=str(Path(calibration).resolve()),
            ranking_freeze=str(out/'RANKING_FREEZE.json'),rank_inputs=reader.files.copy(),base_result=base)
        save(out/'GEOMETRY_JOB.json',job);pending.append(out/'GEOMETRY_JOB.json')
        write_progress(output/'PROGRESS.json',dict(status='ranking',ranked_objects=len(pending),objects=8))
    def finish(job):
        out=job.parent
        with (out/'geometry_worker.log').open('xb')as log:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--geometry-job',str(job)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if proc.returncode:raise RuntimeError('Geometry worker failed: '+str(out/'geometry_worker.log'))
        return json.loads((out/'RESULTS.json').read_text())
    completed={}
    with ThreadPoolExecutor(max_workers=geometry_workers)as pool:
        active={pool.submit(finish,p):p for p in pending}
        for future in as_completed(active):
            result=future.result();completed[result['object_id']]=result
            print('S40_HAND_DOMAIN_GEOMETRY '+json.dumps(result),flush=True)
            write_progress(output/'PROGRESS.json',dict(status='geometry',finished_objects=len(completed),objects=8))
    summary=[completed[obj['object_id']]for obj in objects if obj['object_id']in identities]
    if len(summary) != 8:raise ValueError('Exactly eight predeclared internal frames required')
    reader.verify()
    report=dict(schema='anydex_s40_boya_port_v3_internal_diagnostic',head_recipe=head_recipe,flip_policy='both',
        candidate_budget_boundary='Per-type calibrated mapping and source table support replace DH3 camera cone/common width; top1000 raw, top3500 score, strict .85 and top400 full collisions unchanged. Extra support cost declared.',domain_policy=dict(path=str(Path(domain_policy).resolve()),sha256=policy_sha,policy=policy),status='complete',objects=summary,
        object_count=len(summary),head_queries=queries_total,models=model_metadata,profile='released-dh3-v1',parameters=profile,
        scope='Boya source table prefix and per-type mapping before heads; full cloud after frozen scores; same internal8/32-fit heads, no external or physical success claim',
        dataset_manifest_deserialized=False,labels_used_for_ranking=False,old_ready_queue_read=False,
        new_physics_steps=0,new_CGR_forwards=0,new_training_updates=0,wall_s=time.monotonic()-started,
        no_threshold_tuning=True,geometry_workers=geometry_workers,full_cloud_collision_calls_before_scoring=0,source_sha256={str(p):reader.freeze(p)for p in [Path(__file__).resolve(),BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py']},
        input_sha256=reader.files)
    save(output/'RESULTS.json',report)
    (output/'PROGRESS.json').write_text(json.dumps(dict(status='complete',objects=8))+'\n')
    print(json.dumps(dict(status='complete',objects=len(summary),with_shortlist=sum(x['shortlist']>0 for x in summary),
        with_admitted_geometry=sum(x['geometry_admitted']>0 for x in summary),new_physics_steps=0,wall_s=report['wall_s']),indent=2))
    return report

def main():
    if '--geometry-job'in sys.argv[1:]:
        p=argparse.ArgumentParser();p.add_argument('--geometry-job',type=Path,required=True);return geometry_worker(p.parse_args().geometry_job)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-run',type=Path,required=True);p.add_argument('--training-results',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--domain-policy',type=Path,required=True)
    p.add_argument('--head-recipe',choices=['code100'],required=True)
    p.add_argument('--geometry-workers',type=int,choices=range(1,5),default=4)
    a=p.parse_args();evaluate(a.collection_run.resolve(),a.training_results.resolve(),a.output_dir,a.calibration.resolve(),a.domain_policy.resolve(),a.head_recipe,a.geometry_workers)

if __name__=='__main__':main()
