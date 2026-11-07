#!/usr/bin/env python3
"""Frozen external Boya evaluation: real preactivation-control heads, no tuning.

Generic external scene/protocol manifests supply identities. The unchanged
Boya hand-domain mapping, score expansion and complete geometry worker are
reused. Four 32-fit heads are loaded once on CPU. Ranking is frozen before
full visible-cloud collisions. This stage never plans arms or runs physics.
"""
from __future__ import annotations
import argparse,copy,json,re,sys,time,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
sys.path.insert(0,str(BOYA))
import numpy as np
import torch
from evaluate_boya_decision import Reader,original_inference_class
from deploy_score_first import PROFILES,OFFICIAL_ROBOT_SHA,validate_cached_forward,save,calibrated_slots
from boya_hand_domain import make_hand_queries,expand_hand_queries,SourceHandSupport,validate_policy
from asset_adapter import BoyaAssets
from evaluate_s40_hand_domain import geometry_worker,write_progress

SCHEMA='anydex_boya_external_hand_domain_diagnostic_v1'
MANIFEST_SCHEMA='anydex_boya_external_scene_manifest_v1'
PROTOCOL_SCHEMA='anydex_boya_external_protocol_v1'
TRAINING_SCHEMA='anydex_s40_preactivation_control_v1'
VARIANT='preactivation_selected_mse'
CONFIG=dict(epochs=100,batch_size=128,learning_rate=5e-4,weight_decay=5e-4,optimizer='Adam',scheduler='original PolyLR',poly_power=.99,seed=20261106,input_dim=480,output_dim=20)
GESTURES=(1,6,7,8)
MESH_KEYS={'mesh_sha256','mesh_group_sha256','source_mesh_sha256','source_visual_mesh_sha256','visual_mesh_sha256','collision_mesh_sha256'}


def mesh_shas(document):
    """Collect explicit mesh identities only, never outcome labels or all SHAs."""
    values=set()
    if isinstance(document,dict):
        for key,value in document.items():
            if key in MESH_KEYS:
                if not isinstance(value,str)or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Invalid mesh SHA256: '+key)
                values.add(value)
            elif isinstance(value,(dict,list)):values.update(mesh_shas(value))
    elif isinstance(document,list):
        for value in document:values.update(mesh_shas(value))
    return values


def object_ids(document):
    values=set()
    if isinstance(document,dict):
        if isinstance(document.get('object_id'),str):values.add(document['object_id'])
        for value in document.values():
            if isinstance(value,(dict,list)):values.update(object_ids(value))
    elif isinstance(document,list):
        for value in document:values.update(object_ids(value))
    return values


def ids_from_rows(rows):
    if not isinstance(rows,list)or not rows:raise ValueError('Nonempty frozen object list required')
    ids=[r['object_id']if isinstance(r,dict)else r for r in rows]
    if any(not isinstance(v,str)or not re.fullmatch('[A-Za-z0-9_-]+',v)for v in ids)or len(ids)!=len(set(ids)):raise ValueError('Unsafe or duplicate frozen object identities')
    return ids


def validate_external_manifests(manifest,protocol):
    if manifest.get('schema')!=MANIFEST_SCHEMA or protocol.get('schema')!=PROTOCOL_SCHEMA:raise ValueError('Explicit generic external manifest/protocol schemas required')
    objects=manifest.get('objects');ids=ids_from_rows(objects);approved=manifest.get('approved_object_ids',manifest.get('approved_ids'))
    if approved is None or ids_from_rows(approved)!=ids:raise ValueError('Manifest object order differs from approved identities')
    if 'approved_ids'in manifest and ids_from_rows(manifest['approved_ids'])!=ids:raise ValueError('Conflicting approved identity fields')
    pids=ids_from_rows(protocol.get('objects'))
    if pids!=ids or protocol.get('object_count')!=len(ids):raise ValueError('External protocol identity/order/count mismatch')
    if manifest.get('object_count',len(ids))!=len(ids):raise ValueError('External manifest count mismatch')
    if (protocol.get('arm_budget_per_object')!=4 or protocol.get('physical_attempts_per_object')!=1
        or protocol.get('outcome_dependent_retry')is not False or protocol.get('benchmark_only')is not True
        or protocol.get('training_writeback')is not False or protocol.get('head_recipe')!=VARIANT):raise ValueError('Frozen external top1/no-retry protocol changed')
    for row in objects:
        if not isinstance(row,dict)or not {'object_id','observation_json','representation_dir','mesh_sha256'}<=set(row):raise ValueError('Missing external scene columns')
        if row.get('split','external_validation')!='external_validation':raise ValueError('External scene has a training/internal split')
        if any(row.get(k)is not None for k in ['label','training_label','success','result','trial_outcome']):raise ValueError('External scene manifest must precede grasp outcomes')
        mesh_shas(row)
    if not manifest.get('exclusion_manifest_refs'):raise ValueError('Source-bound training/previously-used mesh exclusion manifests required')
    return objects,ids


def validate_training(training):
    if (training.get('schema')!=TRAINING_SCHEMA or training.get('training_variant')!=VARIANT
        or training.get('status')!='complete' or training.get('inputs_and_sources_unchanged')is not True
        or training.get('external_validation_evaluated')is not False or training.get('all40_refit')is not False
        or training.get('author_original_training')is not False or training.get('config')!=CONFIG
        or set(training.get('internal_heads',{}))!=set(map(str,GESTURES))):raise ValueError('Frozen complete preactivation-control 32-fit training report required')
    groups=training.get('mesh_groups',{})
    if set(groups)!={'train','internal_check'}or len(groups['train'])!=32 or len(groups['internal_check'])!=8 or len(set(groups['train']+groups['internal_check']))!=40:raise ValueError('Require original disjoint32-fit/8-internal groups')
    return set(groups['train']),set(groups['internal_check'])


def validate_checkpoint(checkpoint,gesture,dataset_sha,allowed,internal,external):
    fit=set(checkpoint.get('fit_mesh_sha256',[]))
    if (checkpoint.get('stage')!='internal_check' or checkpoint.get('gesture_id')!=gesture
        or checkpoint.get('training_variant')!=VARIANT or checkpoint.get('author_original_training')is not False
        or checkpoint.get('dataset_manifest_sha256')!=dataset_sha or checkpoint.get('epoch')!=100
        or checkpoint.get('training_config')!=CONFIG or checkpoint.get('input_dim')!=480 or checkpoint.get('output_dim')!=20
        or not fit or not fit<=allowed or fit&internal or fit&external):raise ValueError('Checkpoint is not a disjoint frozen32-fit preactivation head')
    expected=dict(input_num=480,num_multifinger_type=1,num_multifinger_depth=4,two_finger_num_angle=1,two_finger_num_depth=5)
    if checkpoint.get('architecture_kwargs')!=expected:raise ValueError('Original480-to20 inference architecture changed')
    return fit


def validate_scene_binding(obj,observation,reader):
    if observation.get('object_id')!=obj['object_id']:raise ValueError('RGB-D observation belongs to another external object')
    geometry=observation.get('object_geometry',{})
    if geometry.get('source_mesh_sha256')!=obj['mesh_sha256']:raise ValueError('Rendered mesh differs from external scene manifest')
    if not geometry.get('source_mesh'):raise ValueError('Rendered source mesh path required')
    reader.freeze(geometry['source_mesh'],obj['mesh_sha256'])
    for pathkey,shakey in [('mesh_path','mesh_sha256'),('source_visual_mesh','source_visual_mesh_sha256'),('source_mesh','source_mesh_sha256')]:
        if obj.get(pathkey)and obj.get(shakey):reader.freeze(obj[pathkey],obj[shakey])


def evaluate(objects_manifest,protocol_path,training_path,output,calibration,domain_policy,geometry_workers=4):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Fresh external evaluation output required; no overwrite/retry')
    if not 1<=geometry_workers<=4:raise ValueError('Geometry workers must be1..4')
    reader=Reader();started=time.monotonic()
    source_paths=[Path(__file__).resolve(),BOYA/'evaluate_s40_hand_domain.py',BOYA/'deploy_score_first.py',BOYA/'evaluate_boya_decision.py',BOYA/'trial_record.py',BOYA/'asset_adapter.py',BOYA/'build_collection_pool.py',BOYA/'boya_prehead_domain.py',BOYA/'boya_hand_domain.py']
    sources={str(p):reader.freeze(p)for p in source_paths}
    manifest=reader.read(objects_manifest);protocol=reader.read(protocol_path);objects,identities=validate_external_manifests(manifest,protocol)
    protocol_ref=dict(path=str(Path(protocol_path).resolve()),sha256=reader.files[str(Path(protocol_path).resolve())])
    manifest_ref=dict(path=str(Path(objects_manifest).resolve()),sha256=reader.files[str(Path(objects_manifest).resolve())])
    training=reader.read(training_path);allowed,internal=validate_training(training)
    training_ref=dict(path=str(Path(training_path).resolve()),sha256=reader.files[str(Path(training_path).resolve())])
    dataset_ref=training['dataset_manifest'];dataset=reader.read(dataset_ref['path'],dataset_ref['sha256'])
    if dataset.get('mesh_groups')!=training['mesh_groups']or dataset.get('sample_count')!=435:raise ValueError('Frozen435 training manifest metadata mismatch')
    excluded=mesh_shas(dataset)|allowed|internal;excluded_ids=object_ids(dataset);exclusion_refs=[]
    for ref in manifest['exclusion_manifest_refs']:
        doc=reader.read(ref['path'],ref['sha256']);values=mesh_shas(doc)
        if not values:raise ValueError('Historical mesh exclusion manifest contains no explicit mesh identities')
        excluded.update(values);excluded_ids.update(object_ids(doc));exclusion_refs.append(dict(path=str(Path(ref['path']).resolve()),sha256=ref['sha256'],mesh_identities=len(values)))
    external=set().union(*(mesh_shas(row)for row in objects))
    if external&excluded:raise ValueError('External source/prepared mesh overlaps training or previously-used assets')
    if set(identities)&excluded_ids:raise ValueError('External identity was already used for training or evaluation')
    policy=reader.read(domain_policy);validate_policy(policy)
    for p,h in policy['source_sha256'].items():reader.freeze(p,h)
    policy_ref=dict(path=str(Path(domain_policy).resolve()),sha256=reader.files[str(Path(domain_policy).resolve())])
    source=ROOT/'models/minkowski_graspnet_single_point.py';sources[str(source)]=reader.freeze(source);reader.freeze(ROOT/'robot_dh3.py',OFFICIAL_ROBOT_SHA)
    cls=original_inference_class(source,torch);models={};model_metadata={};torch.set_num_threads(4)
    for gesture in GESTURES:
        entry=training['internal_heads'][str(gesture)];path=Path(entry['path']);reader.freeze(path,entry['sha256'])
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        fit=validate_checkpoint(checkpoint,gesture,dataset_ref['sha256'],allowed,internal,external)
        model=cls(**checkpoint['architecture_kwargs']);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval();models[gesture]=model
        model_metadata[str(gesture)]=dict(path=str(path),sha256=entry['sha256'],strict_load=True,training_variant=VARIANT,author_original_training=False,fit_mesh_groups=len(fit),fit_mesh_sha256=sorted(fit),external_mesh_overlap=0)
    profile=copy.deepcopy(PROFILES['released-dh3-v1']);output.mkdir(parents=True)
    assets=BoyaAssets();families=calibrated_slots(assets,reader.read(calibration),reader);support=SourceHandSupport(assets)
    novelty=dict(external_mesh_identities=len(external),excluded_mesh_identities=len(excluded),excluded_object_ids=len(excluded_ids),object_id_intersection=0,intersection=0,exclusion_manifest_refs=exclusion_refs,comparison='Byte-SHA identity for source/prepared geometry; not semantic or geometric equivalence proof',training_manifest_metadata_read=True,training_label_arrays_read=False)
    save(output/'PLAN.json',dict(schema='anydex_boya_external_hand_domain_plan_v1',external_protocol_ref=protocol_ref,objects_manifest_ref=manifest_ref,training_results_ref=training_ref,training_variant=VARIANT,author_original_training=False,inference_activation='Unchanged original final ReLU',domain_policy_ref=policy_ref,parameters=profile,flip_policy='both',approved_object_ids=identities,object_count=len(identities),novelty_audit=novelty,geometry_workers=geometry_workers,external_outcome_labels_read=False,training_label_arrays_read=False,benchmark_only=True,training_writeback=False,new_physics_steps=0,new_training_updates=0))
    pending=[];queries_total=0
    for obj in objects:
        out=output/obj['object_id'];out.mkdir();obs_path=Path(obj['observation_json']).resolve();obs=reader.read(obs_path);validate_scene_binding(obj,obs,reader)
        frame=Path(obj['representation_dir']).resolve();receipt=reader.read(frame/'RESULTS.json');validate_cached_forward(obs_path,obs,frame,receipt,reader);reader.freeze(calibration)
        with np.load(frame/'raw_camera_candidates.npz',allow_pickle=False)as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
        queries,inputs,indices,domain=make_hand_queries(raw,features,obs['T_WC'],policy,profile,families,assets,dict(domain_policy=policy_ref,observation=dict(path=str(obs_path),sha256=reader.files[str(obs_path)]),raw_npz=dict(path=str(frame/'raw_camera_candidates.npz'),sha256=reader.files[str(frame/'raw_camera_candidates.npz')]),policy_sources=policy['source_sha256']),support=support)
        save(out/'DOMAIN.json',domain)
        for gesture,model in models.items():
            ids=[i for i,q in enumerate(queries)if q['gesture_id']==gesture]
            with torch.inference_mode():
                for start in range(0,len(ids),128):
                    group=ids[start:start+128];pred,_=model(torch.from_numpy(np.stack([inputs[i]for i in group])))
                    if tuple(pred.shape)!=(len(group),1,20)or not torch.isfinite(pred).all():raise ValueError('Wrong/nonfinite original head output')
                    for i,row in zip(group,pred[:,0].numpy()):queries[i]['all_head_outputs20']=row.tolist()
        queries_total+=len(queries);ranked,shortlist=expand_hand_queries(queries,profile)
        np.savez_compressed(out/'head_outputs.npz',outputs20=np.asarray([q['all_head_outputs20']for q in queries],np.float32).reshape(-1,20),source_indices=np.asarray([q['source_candidate_index']for q in queries]),gesture_ids=np.asarray([q['gesture_id']for q in queries]),raw_depth_indices=np.asarray([q['raw_depth_index']for q in queries]),if_flip=np.asarray([q['if_flip']for q in queries]))
        save(out/'shortlist.json',dict(candidates=shortlist,profile='released-dh3-v1',parameters=profile,original_source_indices=indices,flip_policy='both',evaluation_role='external_validation',scope='Frozen unseen external scenes; no outcome labels or previous ready queues used'))
        reader.verify()
        save(out/'RANKING_FREEZE.json',dict(outputs_sha256={str(p):reader.freeze(p)for p in [out/'DOMAIN.json',out/'head_outputs.npz',out/'shortlist.json']},ranking_inputs_sha256=copy.deepcopy(reader.files),models=model_metadata,external_protocol_ref=protocol_ref,training_results_ref=training_ref,training_variant=VARIANT,evaluation_role='external_validation',labels_used_for_ranking=False,old_admitted_queue_read=False,dataset_manifest_deserialized=True,training_label_arrays_read=False,geometry_calls_before_freeze=0,arm_calls_before_freeze=0,geometry_calls_definition='Full visible-point-cloud collision calls; exact source table prefix counted separately',query_count_definitions=dict(geometry_calls_before_freeze='full_cloud_collision_calls',support_plane_queries_before_freeze='open_hand_table_support_function_queries_not_full_collision_admission'),support_plane_queries_before_freeze=domain['counts']['support_plane_tests'],support_plane_is_full_collision_admission=False,support_plane_predicate='Original .0005 table predicate, all fingers and pregrasp endpoint, unchanged'))
        scores=[q['learned_score']for q in ranked]
        base=dict(object_id=obj['object_id'],evaluation_role='external_validation',domain_counts=domain['counts'],raw_CGR_topk=len(indices),head_queries=len(queries),expanded_combinations=len(ranked),shortlist=len(shortlist),maximum_score=max(scores)if scores else None,score_is_calibrated_probability=False,arm_checked=0,selected=None,new_physics_steps=0,new_CGR_forwards=0,threshold_changed=False,old_limited_queue_used=False)
        job=dict(output_dir=str(out),frame_dir=str(frame),observation=str(obs_path),calibration=str(Path(calibration).resolve()),ranking_freeze=str(out/'RANKING_FREEZE.json'),rank_inputs=reader.files.copy(),base_result=base)
        save(out/'GEOMETRY_JOB.json',job);pending.append(out/'GEOMETRY_JOB.json');write_progress(output/'PROGRESS.json',dict(status='ranking',ranked_objects=len(pending),objects=len(identities)))
    def finish(job):
        with (job.parent/'geometry_worker.log').open('xb')as log:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--geometry-job',str(job)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        if proc.returncode:raise RuntimeError('Geometry worker failed: '+str(job.parent/'geometry_worker.log'))
        return json.loads((job.parent/'RESULTS.json').read_text())
    completed={}
    with ThreadPoolExecutor(max_workers=geometry_workers)as pool:
        active={pool.submit(finish,p):p for p in pending}
        for future in as_completed(active):
            result=future.result();completed[result['object_id']]=result;print('BOYA_EXTERNAL_GEOMETRY '+json.dumps(result),flush=True);write_progress(output/'PROGRESS.json',dict(status='geometry',finished_objects=len(completed),objects=len(identities)))
    summary=[completed[oid]for oid in identities]
    if len(summary)!=len(identities):raise ValueError('External frozen denominator changed')
    reader.verify()
    report=dict(schema=SCHEMA,status='complete',evaluation_role='external_validation',external_protocol_ref=protocol_ref,objects_manifest_ref=manifest_ref,training_results_ref=training_ref,head_recipe=VARIANT,training_variant=VARIANT,author_original_training=False,original_inference_activation_unchanged=True,flip_policy='both',candidate_budget_boundary='Unchanged per-type calibration/source table domain; top1000 raw/top3500 score/strict.85/top400 full cloud; arm4 and one actual attempt handled by next stage',domain_policy=dict(**policy_ref,policy=policy),objects=summary,object_count=len(summary),approved_object_ids=identities,head_queries=queries_total,models=model_metadata,profile='released-dh3-v1',parameters=profile,novelty_audit=novelty,scope='Frozen external scene test of declared Boya preactivation-training adaptation; score/geometry stage only, no physical success claim and not the full original paper protocol',dataset_manifest_deserialized=True,training_label_arrays_read=False,labels_used_for_ranking=False,old_ready_queue_read=False,benchmark_only=True,training_writeback=False,new_physics_steps=0,new_CGR_forwards=0,new_training_updates=0,no_threshold_tuning=True,geometry_workers=geometry_workers,full_cloud_collision_calls_before_scoring=0,source_sha256=sources,input_sha256=reader.files,inputs_unchanged=True,wall_s=time.monotonic()-started)
    save(output/'RESULTS.json',report);write_progress(output/'PROGRESS.json',dict(status='complete',objects=len(identities)))
    print(json.dumps(dict(status='complete',objects=len(summary),with_shortlist=sum(r['shortlist']>0 for r in summary),with_admitted_geometry=sum(r['geometry_admitted']>0 for r in summary),new_physics_steps=0,wall_s=report['wall_s']),indent=2));return report


def main():
    if '--geometry-job'in sys.argv[1:]:
        p=argparse.ArgumentParser();p.add_argument('--geometry-job',type=Path,required=True);return geometry_worker(p.parse_args().geometry_job)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--objects-manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--training-results',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--domain-policy',type=Path,required=True);p.add_argument('--geometry-workers',type=int,choices=range(1,5),default=4)
    a=p.parse_args();evaluate(a.objects_manifest,a.protocol,a.training_results,a.output_dir,a.calibration,a.domain_policy,a.geometry_workers)
if __name__=='__main__':main()
