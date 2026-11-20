"""Score frozen audited Boya outcomes and inspect shared feasible candidate pools."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import author_collection_batch as B
import author_training as T
from author_mjx_policy_compare_worker import ranking_order
from author_mjx_shared_pair_worker import Scorer


OUT = B.ROOT/'outputs'
MAINLINE = OUT/'boya_mjx_mainline_status_20261120_v1/CHECK.json'
FORMAL = OUT/'boya_mjx_cgr_to100_20261119_v1/collection/PROTOCOL.json'
DIAGNOSIS = OUT/'boya_mjx_head_diagnosis_20261119_v1'
SHARED = OUT/'boya_mjx_shared_pair_20261119_v1'
COVERAGE = OUT/'boya_mjx_shared_pool_coverage_20261119_v1/FINAL_REVIEW.json'
# Enrol in this order and keep the first eligible result of identical layouts.
CAMPAIGNS = (
    'boya_mjx_actual_one_hour_20261118_v1',
    'boya_mjx_continuous_queue_control_20261118_v1',
    'boya_mjx_cgr_fallback_20261119_v2',
    'boya_mjx_cgr_empty_pool_rescue_20261119_v1',
    'boya_mjx_policy_compare_20261119_v1',
    SHARED.name,
)


def layout_id(scene_file):
    scene = B.read(scene_file)
    poses = sorted(
        (item['object_id'], item['source_mesh_sha256'], item['scale'], item['world_T_object'])
        for item in scene['objects']
    )
    return hashlib.sha256(json.dumps(poses, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_layouts(npz):
    paths = {}
    layouts, ids = set(), set()
    for path, digest in zip(npz['record_json'],npz['record_sha256'],strict=True):
        assert B.sha(path)==str(digest)
        record = B.read(path)
        protocol = record['domain']['path']
        if protocol not in paths:
            B.verify({'domain':record['domain']})
            paths[protocol] = {row['scene_id']:row['scene'] for row in B.read(protocol)['rows'] if row['scene']}
        refs = paths[protocol][record['scene_id']]
        B.verify({'scene':refs})
        layouts.add(layout_id(refs['path']))
        ids.add(record['scene_id'])
    return layouts,ids


def collect_archived(corpus, formal, seen_layouts, seen_ids, fit_objects):
    directory = OUT/corpus/'collection'
    files = {key:B.binding(directory/name) for key,name in (('protocol','PROTOCOL.json'),('run','RESULTS.json'),('audit','READ_ONLY_AUDIT.json'))}
    p = B.read(files['protocol']['path'])
    raw = B.read(files['run']['path'])
    audit = B.read(files['audit']['path'])
    assert audit['status']=='read_only_audit_pass' and audit['original_result']==files['run']
    assert p['domain']==formal['domain']
    for key in ('control_dt_s','source_dt_s','GPU_noslip_iterations','GPU_ccd_tolerance_m'):
        assert p[key]==formal[key]
    for key in ('backend','calibration','cloud_manifest','template:0','checkpoint'):
        assert p['inputs'][key]==formal['inputs'][key]
    audited = {item['index']:item for item in audit['rounds']}
    output, exclusions = [],Counter()
    for spec,record in zip(p['rows'],raw['records'],strict=True):
        if not record['eligible']:
            continue
        row = audited[spec['index']]
        assert all(row[key] for key in ('eligible','original_controls_exact','return_pass','CGR_input_and_receipt_exact'))
        assert row['label']==record['label'] and record['label'] in (0,1)
        assert record['backend']=='MJX-Warp' and record['gesture_id']==1 and record['exclude_training']
        refs = record['bindings']
        B.verify({key:refs[key] for key in ('candidate','observation','scene','execution','commands')})
        B.verify({'layout':spec['scene']})
        candidate = B.read(refs['candidate']['path'])
        if candidate.get('recovery_variant'):
            exclusions['changed_mapping'] += 1
            continue
        identity = layout_id(spec['scene']['path'])
        if identity in seen_layouts or record['scene_id'] in seen_ids:
            exclusions['training_selection_or_duplicate_layout'] += 1
            continue
        seen_layouts.add(identity)
        seen_ids.add(record['scene_id'])
        actual = B.read(refs['execution']['path'])
        assert actual['status']=='complete' and actual['stop'] is None
        assert record['label']==int(actual['evaluation']['task_success'])
        assert set(spec['source_object_ids'])==set(B.read(refs['observation']['path'])['object_ids'])
        has_fit_body = bool(set(spec['source_object_digests'].values())&fit_objects)
        assert not has_fit_body or spec['split']=='fit'
        output.append(dict(campaign=corpus,index=spec['index'],scene_id=record['scene_id'],layout_sha256=identity,
            split=spec['split'],fit_body_overlap=has_fit_body,label=record['label'],objects=spec['source_object_ids'],
            candidate=refs['candidate'],execution=refs['execution'],observation=refs['observation']))
    return output,dict(files=files,included=len(output),excluded=dict(exclusions))


def freeze(root):
    root.mkdir(parents=True,exist_ok=False)
    M = B.read(MAINLINE)
    B.verify({'data':M['dataset'],'training':M['formal_training']})
    with np.load(M['dataset']['path'],allow_pickle=False) as f:
        train,check = T.validate_arrays({key:f[key] for key in f.files})
        assert len(train)+len(check)==117
        seen_layouts,seen_ids=training_layouts(f)
        fit_objects=set(f['scene_mesh_sha256'][train].ravel())
    forbidden_layouts = seen_layouts.copy()
    formal = B.read(FORMAL)
    rows, receipts = [],{}
    for corpus in CAMPAIGNS:
        enrolled, receipts[corpus] = collect_archived(corpus,formal,seen_layouts,seen_ids,fit_objects)
        rows.extend(enrolled)
    assert len(rows)==len({r['layout_sha256'] for r in rows}) and rows
    shared = SHARED/'collection'
    shared_protocol = B.read(shared/'PROTOCOL.json')
    checked = B.read(COVERAGE)
    assert checked['status']=='offline_coverage_complete'
    B.verify({'comparison':checked['source_comparison']})
    known_plans = {int(item['pair']):item for item in checked['candidate_pools']}
    pools = []
    for spec in shared_protocol['rows'][::2]:
        index = spec['pair_index']
        directory = shared/'pairs'/f'{index:03d}'
        if not (directory/'ADMISSION.json').exists() or layout_id(spec['scene']['path']) in forbidden_layouts:
            continue
        admission = B.read(directory/'ADMISSION.json')
        B.verify({'rank':admission['ranking']})
        ranking = B.read(admission['ranking']['path'])
        B.verify({'pool':ranking['shared_pool'],'observation':ranking['observation']})
        checked_pool = B.read(ranking['shared_pool']['path'])['candidates']
        assert [x['candidate_sha256'] for x in checked_pool]==ranking['hashes']
        cgr_scores = [candidate['score'] for candidate in checked_pool]
        assert all(a >= b for a, b in zip(cgr_scores, cgr_scores[1:]))
        ready_ranks = None
        if index in known_plans:
            plan = known_plans[index]
            assert plan['candidates']==len(checked_pool)==len(plan['plans'])
            for item in plan['plans']:
                B.verify({'plan':item['plan']})
                assert checked_pool[item['rank']]['candidate_sha256']==item['candidate_sha256']
                assert B.read(item['plan']['path'])['status']==item['status']
            ready_ranks = [a['rank'] for a in plan['plans'] if a['status']=='ready']
            assert sorted(ready_ranks)==sorted(plan['plan_ready_ranks'])
        pools.append(dict(pair=index,draw=spec['draw'],pool=ranking['shared_pool'],frame=ranking['observation'],ready_ranks=ready_ranks))
    head=B.read(M['formal_training']['path'])['inference_check']['checkpoint']
    diagnostic=B.binding(DIAGNOSIS/'linear_output/final_diagnostic.pt')
    B.verify({'formal':head,'diagnostic':diagnostic})
    policy=dict(status='frozen_before_scoring',formal_model=head,diagnostic_model=diagnostic,training_source=B.binding(MAINLINE),
        selection='All eligible archived runs in listed order; exact source placement de-duplication; exclude training and model-selection layouts',
        independent_layouts=rows,archive_audits=receipts,shared_candidate_pools=pools,planning_coverage=B.binding(COVERAGE),
        score_threshold=.5,primary_model='formal100',ranking_baseline='raw CGR score',
        no_new_physics=True,no_new_training=True)
    B.save(root/'PROTOCOL.json',policy,exclusive=True)
    print(dict(status=policy['status'],cases=len(rows),labels=dict(Counter((r['split']+'_'+str(r['label'])) for r in rows)),
               pools=len(pools),exclusions={key:item['excluded'] for key,item in receipts.items()}),flush=True)


def metrics(labels,values):
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    assert labels.shape == values.shape and np.isfinite(values).all()
    positives, negatives = values[labels == 1], values[labels == 0]
    if not len(positives):
        return dict(roc_auc=None, average_precision=None)

    auc = None
    if len(negatives):
        differences = positives[:, None] - negatives[None, :]
        auc = float(np.mean((differences > 0) + .5 * (differences == 0)))

    order = np.argsort(-values, kind='stable')
    # Evaluate precision once per score threshold, including every tied row.
    boundaries = np.r_[np.flatnonzero(np.diff(values[order])), len(order) - 1]
    true_positives = np.cumsum(labels[order])[boundaries]
    precision = true_positives / (boundaries + 1)
    newly_recalled = np.diff(np.r_[0, true_positives])
    average_precision = float(np.sum(precision * newly_recalled / len(positives)))
    return dict(roc_auc=auc, average_precision=average_precision)


def formal_score(candidates,model):
    if not candidates:
        return np.empty(0,dtype=np.float32)
    features=np.array([x['canonical_cgr480'] for x in candidates],dtype=np.float32)
    addresses=np.array([4*x['raw_depth_index']+x['extra_depth_index'] for x in candidates],dtype=np.int64)
    assert features.shape==(len(candidates),480) and np.isfinite(features).all() and ((addresses>=0)&(addresses<20)).all()
    with torch.no_grad():
        scores=model(torch.from_numpy(features))[0].numpy()
    assert scores.shape==(len(candidates),1,20) and np.isfinite(scores).all()
    return scores[np.arange(len(candidates)),0,addresses]


def evaluate(root):
    p=B.read(root/'PROTOCOL.json')
    assert p['status']=='frozen_before_scoring'
    B.verify({key:p[key] for key in ('formal_model','diagnostic_model','training_source','planning_coverage')})
    torch.set_num_threads(1)
    model = T.components()[1]().eval()
    checkpoint = torch.load(p['formal_model']['path'], map_location='cpu', weights_only=True)
    assert checkpoint['schema'] == 'anydex_boya_safe_author_head_v1' and checkpoint['gesture_id'] == 1
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    diagnostic=Scorer(p['diagnostic_model'])
    rows=p['independent_layouts']
    B.verify({f'case:{i}:{key}':row[key] for i,row in enumerate(rows) for key in ('candidate','execution','observation')})
    cand=[B.read(row['candidate']['path']) for row in rows]
    for row,model_score,diagnostic_score,c in zip(rows,formal_score(cand,model),diagnostic.scores(cand),cand,strict=True):
        row.update(formal_score=float(model_score),diagnostic_score=float(diagnostic_score),cgr_score=c['score'])
    info={}
    for split in ('all','fit','internal'):
        sample=[r for r in rows if split=='all' or r['split']==split]
        truth=np.asarray([a['label'] for a in sample],dtype=bool)
        scores={k:np.asarray([a[k] for a in sample]) for k in ('formal_score','diagnostic_score','cgr_score')}
        pred=scores['formal_score']>=p['score_threshold']
        info[split]=dict(rows=len(sample),positive=int(truth.sum()),negative=int((~truth).sum()),
            formal_confusion=dict(tp=int((truth&pred).sum()),fp=int((~truth&pred).sum()),
                                  tn=int((~truth&~pred).sum()),fn=int((truth&~pred).sum())),
            scores={key:metrics(truth,values) for key,values in scores.items()})
    pool_rows=[]
    for row in p['shared_candidate_pools']:
        B.verify({'pool':row['pool'],'frame':row['frame']})
        options=B.read(row['pool']['path'])['candidates']
        order=ranking_order(formal_score(options,model)).tolist()
        ready=row['ready_ranks']
        pool_rows.append(dict(pair=row['pair'],draw=row['draw'],candidates=len(options),
            formal_changes_top1=bool(order and order[0]!=0),complete_kinematic_coverage=ready is not None,
            feasible_candidate_count=len(ready) if ready is not None else None,
            formal_first_feasible=next((i for i in order if i in ready),None) if ready is not None else None,
            cgr_first_feasible=min(ready) if ready else None))
    report=dict(status='read_only_outcome_and_pool_eval_pass',protocol=B.binding(root/'PROTOCOL.json'),
        outcome_metrics=info,examples=rows,shared_candidate_pools=pool_rows,
        pool_summary=dict(observations=len(pool_rows),multi_candidate=sum(r['candidates']>=2 for r in pool_rows),
            complete_kinematic_coverage=sum(r['complete_kinematic_coverage'] for r in pool_rows),
            multi_feasible=sum((r['feasible_candidate_count'] or 0)>=2 for r in pool_rows),
            formal_changes_top1=sum(r['formal_changes_top1'] for r in pool_rows),
            formal_changes_feasible_choice=sum(r['formal_first_feasible']!=r['cgr_first_feasible'] for r in pool_rows)),
        paired_execution_gain_measured=False,additional_physics_steps=0,training_updates=0)
    B.save(root/'RESULTS.json',report,exclusive=True)
    print(dict(outcomes=info,pools=report['pool_summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','evaluate'))
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    {'freeze':freeze,'evaluate':evaluate}[args.action](args.root.resolve())
