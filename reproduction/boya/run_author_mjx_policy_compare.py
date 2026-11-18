"""Finite paired new-layout CGR-vs-pilot-head execution, without video."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import author_collection_batch as B
import author_collection_scenes as S
import author_collection_wave_resources as R
import author_mjx_batch_collection_v9 as F
from author_mjx_full_comparison_v4 import dispatch

SEED=2026111901
CATALOG=Path('/mnt/public/datasets/AnyDexGrasp/boya_collection_20261114_v1/manual_start_20261116_v2/scene_queue/CATALOG.json')
ENTRY=B.HERE/'author_mjx_policy_compare_collection.py'
PILOT=B.ROOT/'outputs/boya_mjx_head_pilot_20261118_v1'


def freeze(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    pilot=B.read(PILOT/'TRAINING_RESULTS.json');head=pilot['inference_checkpoint']
    if pilot['status']!='limited_data_training_and_inference_complete' or pilot['optimizer_updates']!=100:
        raise ValueError('One fixed completed pilot required')
    B.verify({'checkpoint':head});catalog=B.read(CATALOG)
    S.verify(catalog['input_sha256']);S.verify(catalog['source_sha256'])
    intent=dict(schema='anydex_mjx_paired_policy_comparison_v1',seed=SEED,
        fixed_new_internal_scene_draws=16,maximum_ready_layout_pairs=8,maximum_physical_attempts=16,
        placement_selection='First at most8 placement-ready layouts among fixed16 draws; no CGR/IK/label selection',
        policies=['cgr','head'],ranking='CGR top1 versus fixed-head reranking of unchanged collision-filtered CGR top10',
        candidate_fallback=False,physics_retries=0,training_updates=0,video_recording=False,
        independent_new_layouts=True,objects_from_existing_internal_split=True,new_unseen_object_domain=False,
        shared_initial_source_layout=True,independent_live_settling_and_photos=True,
        candidate_network_rng='Original persistent engine restores the same post-initialization RNG for every frame',
        exact_network_input_and_pool_match_must_be_measured=True,
        no_claim_of_identical_physical_trajectories=True,head=head,pilot_results=B.binding(PILOT/'TRAINING_RESULTS.json'))
    B.save(root/'INTENT.json',intent,exclusive=True)
    draws=[];chosen=[];cache={}
    for index in range(16):
        spec=S.sample_scene(catalog,'internal',index,SEED,geometry_cache=cache);spec['catalog']=B.binding(CATALOG)
        path=root/'queue/specs'/f'{index:02d}.json';B.save(path,spec,exclusive=True)
        draws.append(dict(index=index,status=spec['status'],scene_id=spec['scene_id'],scene_spec=B.binding(path)))
        if spec['status']=='ready' and len(chosen)<8:chosen.append((index,path,spec))
    if not chosen:raise ValueError('Fixed draws produced no ready layouts; do not silently redraw')
    with np.load(B.read(PILOT/'PROTOCOL.json')['inputs']['dataset']['path'],allow_pickle=False)as z:
        old_ids=z['trial_id'].tolist()
    rows=[];rng=np.random.default_rng(SEED)
    for pair,(source_index,path,spec) in enumerate(chosen):
        if any(spec['scene_id'] in v for v in old_ids):raise ValueError('Comparison layout leaked into fit/selection')
        scene=S.build_scene(path,root/'queue/build'/f'{pair:02d}')
        policy_order=['cgr','head'] if rng.integers(2)==0 else ['head','cgr']
        for policy in policy_order:
            value=copy.deepcopy(scene);value['paired_source_scene_id']=scene['scene_id']
            value['scene_id']+=f'_compare_{policy}'
            dest=root/'queue/scenes'/f'{len(rows):02d}.json';B.save(dest,value,exclusive=True)
            rows.append(dict(index=len(rows),pair_index=pair,comparison_policy=policy,source_draw_index=source_index,
                scene_id=value['scene_id'],split='internal',status='ready',scene=B.binding(dest),
                scene_spec=B.binding(path),source_object_ids=spec['source_object_ids'],source_object_digests=spec['source_object_digests'],
                origin='new_internal_layout_paired_source_coordinates_no_label_selection'))
    B.save(root/'DRAWS.json',dict(draws=draws,selected_source_indices=[i for i,_,_ in chosen]),exclusive=True)
    queue=root/'QUEUE_SOURCE.json';B.save(queue,dict(rows=rows),exclusive=True)
    base=F.freeze(root/'base_freeze',environments=16,source_protocol=queue)
    directory=root/'collection';directory.mkdir();p=copy.deepcopy(base)
    p.update(schema='anydex_mjx_paired_policy_comparison_v1',seed=SEED,comparison_only=True,
        pair_count=len(chosen),video_recording=False,exclude_training=True,formal_production_quota_frozen=False,
        formal_training_eligible=False,comparison_underfilled=len(rows)<16,training_updates=0)
    p['env']['ANYDEX_CONTACT_SPOOL_ROOT']=str(directory/'contact_spool')
    p['inputs'].update(comparison_head=head,comparison_pilot_results=B.binding(PILOT/'TRAINING_RESULTS.json'),
        comparison_catalog=B.binding(CATALOG),comparison_intent=B.binding(root/'INTENT.json'),comparison_draws=B.binding(root/'DRAWS.json'),
        base_protocol=B.binding(root/'base_freeze/PROTOCOL.json'))
    for row in rows:p['inputs']['scene_spec:'+str(row['index'])]=row['scene_spec']
    for name in ('author_mjx_policy_compare_worker.py',ENTRY.name,Path(__file__).name,'author_training.py',
        'author_mjx_hour_collection.py','author_hour_audit.py','author_hour_metrics.py','author_expansion_progress.py','audit_author_mjx_wave_snapshot.py'):
        p['sources'][name]=B.binding(B.HERE/name)
    archive=directory/'source_archive';archive.mkdir()
    for name,binding in p['sources'].items():shutil.copyfile(binding['path'],archive/name)
    B.save(directory/'PROTOCOL.json',p,exclusive=True)
    shutil.copytree(B.ROOT/'outputs/boya_mjx_fresh_expansion_20261118_v1/collection/warp_cache',directory/'warp_cache')
    result=dict(status='frozen_not_started',new_draws=16,placement_nulls=sum(x['status']!='ready'for x in draws),
        paired_layouts=len(chosen),policy_slots=len(rows),environments=16,
        max_native_environment_steps=p['max_native_environment_steps'],protocol=B.binding(directory/'PROTOCOL.json'),
        video_recording=False,training_updates=0)
    B.save(root/'COMPARISON_READY.json',result,exclusive=True);print(result,flush=True);return result


def summarize(root):
    from author_mjx_policy_compare_worker import Ranker,ranking_order,semantic_candidate
    root=Path(root).resolve();directory=root/'collection';p=B.read(directory/'PROTOCOL.json')
    B.verify(p['inputs']);B.verify(p['sources'])
    result=B.read(directory/'RESULTS.json');audit=B.read(directory/'READ_ONLY_AUDIT.json')
    if result['status']!='complete' or audit['status']!='read_only_audit_pass' or audit['original_result']!=B.binding(directory/'RESULTS.json'):
        raise ValueError('Complete original native audit required')
    ranker=Ranker(p['inputs']['comparison_head']);checked={r['index']:r for r in audit['rounds']}
    groups={};counts={policy:dict(assigned=0,executed=0,eligible=0,success=0,failure=0,null=0)for policy in ['cgr','head']}
    for spec in p['rows']:
        d=directory/'scenes'/f"{spec['index']:02d}";record=B.read(d/'RECORD.json');policy=spec['comparison_policy']
        item=dict(index=spec['index'],policy=policy,status=record['status'],label=record['label'],eligible=record['eligible'])
        group=groups.setdefault(spec['pair_index'],dict(pair=spec['pair_index'],source_draw_index=spec['source_draw_index'],policies={}))
        groups[spec['pair_index']]['policies'][policy]=item
        c=counts[policy];c['assigned']+=1;c['eligible']+=int(record['eligible']);c['success']+=int(record['label']==1)
        c['failure']+=int(record['label']==0);c['null']+=int(record['label']is None);c['executed']+=int((d/'execution/RESULTS.json').exists())
        if record['eligible'] and not checked[spec['index']].get('return_pass'):raise ValueError('Eligible trial lacks audited return')
        receipt=d/'contact/RANKING_RECEIPT.json'
        if receipt.exists():
            saved=B.read(receipt);pool=B.read(d/'contact/ORIGINAL_POOL.json');candidates=pool['candidates']
            scores=ranker.scores(candidates);np.testing.assert_array_equal(scores,np.asarray(saved['selected_output_scores'],np.float32))
            expected=list(range(len(scores))) if policy=='cgr' else ranking_order(scores).tolist()
            if expected!=saved['applied_order'] or saved['checkpoint']!=p['inputs']['comparison_head']:
                raise ValueError('Policy or fixed checkpoint changed')
            if saved['semantic_original_pool']!=[semantic_candidate(c)for c in candidates]:raise ValueError('Original semantic pool changed')
            prepared=B.read(d/'contact/pipeline/collection/PREPARED.json')
            if [c['candidate_sha256']for c in prepared['candidates']]!=[candidates[i]['candidate_sha256']for i in expected]:
                raise ValueError('Head ranking was not applied to the original unchanged candidates')
            item.update(selected_original_rank=saved['selected_original_rank'],ranking_receipt=B.binding(receipt),
                semantic_pool=saved['semantic_original_pool'],candidate_count=len(candidates))
            if record['bindings'].get('candidate'):
                chosen=B.read(record['bindings']['candidate']['path'])
                if chosen['candidate_sha256']!=saved['selected_candidate_sha256']:raise ValueError('Executed candidate differs from ranked choice')
        rep=d/'representation/RESULTS.json'
        if rep.exists():
            value=B.read(rep);item['input_points_sha256']=value['input_points_sha256'];item['input_coords_sha256']=value['input_coords_sha256']
            with np.load(d/'representation/raw_camera_candidates.npz',allow_pickle=False)as z:
                item['_raw_candidates']=z['candidates'].copy();item['_raw_features']=z['candidate_features'].copy()
    for group in groups.values():
        a,b=group['policies']['cgr'],group['policies']['head']
        group['network_input_exact']=all(a.get(k) is not None and a.get(k)==b.get(k)for k in ['input_points_sha256','input_coords_sha256'])
        group['raw_candidates_and_features_exact']=all(k in a and k in b and np.array_equal(a[k],b[k])for k in ['_raw_candidates','_raw_features'])
        group['semantic_original_pool_exact']=('semantic_pool'in a and 'semantic_pool'in b and a['semantic_pool']==b['semantic_pool'])
        group['both_qualified']=a['eligible'] and b['eligible']
        group['head_task_win']=b['label']==1 and a['label']!=1
        group['cgr_task_win']=a['label']==1 and b['label']!=1
        if group['semantic_original_pool_exact'] and a.get('candidate_count',0):
            group['same_selected_candidate']=b.get('selected_original_rank')==0
        for item in [a,b]:
            item.pop('_raw_candidates',None);item.pop('_raw_features',None);item.pop('semantic_pool',None)
    pairs=list(groups.values())
    output=dict(status='audited_paired_policy_comparison_complete',counts=counts,pairs=pairs,
        policy_definition='Original CGR top1 versus fixed pilot head-only reranking of original collision-filtered CGR top10',
        exact_network_input_pairs=sum(g['network_input_exact']for g in pairs),
        exact_raw_candidate_and_feature_pairs=sum(g['raw_candidates_and_features_exact']for g in pairs),
        exact_semantic_pool_pairs=sum(g['semantic_original_pool_exact']for g in pairs),
        both_qualified_pairs=sum(g['both_qualified']for g in pairs),head_task_wins=sum(g['head_task_win']for g in pairs),
        cgr_task_wins=sum(g['cgr_task_win']for g in pairs),
        head_checkpoint=p['inputs']['comparison_head'],source_audit=B.binding(directory/'READ_ONLY_AUDIT.json'),
        native_environment_steps=result['world']['native_environment_integrations'],
        paired_source_layouts=True,independent_live_physics_observations=True,
        exact_trajectory_equivalence_claimed=False,small_exploratory_comparison=True,
        video_recording=False,training_updates=0,exclude_training=True)
    B.save(root/'POLICY_COMPARISON_RESULTS.json',output,exclusive=True);return output


def run(root):
    root=Path(root).resolve();ready=B.read(root/'COMPARISON_READY.json');B.verify({'protocol':ready['protocol']})
    if (root/'PIPELINE_STARTED.json').exists():raise ValueError('No dispatched comparison retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=B.ROOT).strip():raise ValueError('Clean worktree required')
    B.save(root/'PIPELINE_STARTED.json',dict(process=R.process_identity(__import__('os').getpid()),unix_s=time.time(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B.ROOT,text=True).strip()),exclusive=True)
    began=time.monotonic();stages={}
    try:
        stages['collect_and_audit']=dispatch([sys.executable,'-B',str(ENTRY),'--root',str(root/'collection')],root/'timing/collect_and_audit')
        result=summarize(root);R.assert_quiescent(root/'collection')
        B.save(root/'PIPELINE_FINISHED.json',dict(status='complete',wall_s=time.monotonic()-began,stages=stages,
            comparison=B.binding(root/'POLICY_COMPARISON_RESULTS.json'),video_recording=False,training_updates=0),exclusive=True)
        print({k:result[k]for k in ['status','counts','exact_network_input_pairs','exact_semantic_pool_pairs','head_task_wins','cgr_task_wins']},flush=True)
    except BaseException:
        import traceback
        B.save(root/'PIPELINE_FAILURE.json',dict(error=traceback.format_exc(),stages=stages,wall_s=time.monotonic()-began),exclusive=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    if args.action=='freeze':freeze(args.root)
    else:run(args.root)
