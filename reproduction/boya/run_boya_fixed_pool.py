"""Exhaust a frozen finite Boya pool to separate coverage and ranking losses.

Only unexecuted, unique poses are dispatched. Historical trials are reused
only under the identical controlled-wrist executor and physical conditions.
No model is trained and no physics setting, candidate or old receipt is changed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'reproduction/boya'
OBJECTS = ('egad_A3','egad_A6','egad_E0')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, value):
    p=Path(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


def check_hashes(sections):
    for section in sections:
        for path,digest in section.items():
            if sha(path)!=digest:
                raise ValueError('Frozen file changed: '+path)


def planned_queue(candidates):
    groups={oid:[c for c in candidates if c['object_id']==oid and c['requires_new_physical_attempt']] for oid in OBJECTS}
    return [groups[oid][i]['candidate_id'] for i in range(max(map(len,groups.values())))
            for oid in OBJECTS if i<len(groups[oid])]


def prepare(out, catalog_path):
    from fixed_pool_ranking import score_catalog
    if out.exists():
        raise ValueError('Use a fresh output; never overwrite a prepared experiment')
    catalog_path=Path(catalog_path).resolve()
    catalog=read(catalog_path)
    assert catalog['complete'] and catalog['total_unique']==57 and catalog['new_attempts']==42
    assert catalog['reusable_unique']==15 and catalog['near_not_exact_duplicates']==[]
    assert {o:sum(c['object_id']==o for c in catalog['candidates']) for o in OBJECTS}==dict(zip(OBJECTS,(31,11,15)))
    out.mkdir(parents=True)
    start=time.monotonic()
    save(out/'CATALOG.json',catalog)
    sources={}
    inputs={str(catalog_path):sha(catalog_path)}
    inputs.update(catalog['input_sha256'])
    old=ROOT/'outputs/egad_floating_three_20261108_v1/protocol.json'
    prior=read(old)
    sources.update(prior['source_sha256'])
    inputs.update(prior['input_sha256'])
    inputs[str(old)]=sha(old)
    # Protect the complete old 435 records and eight original/preactivation heads.
    old_data=ROOT/'outputs/boya_shared_score_20261111_v2/PROTOCOL.json'
    inputs.update(read(old_data)['input_sha256'])
    inputs[str(old_data)]=sha(old_data)
    for name in ('run_boya_fixed_pool.py','fixed_pool_ranking.py','audit_boya_fixed_pool.py',
                 'publish_boya_fixed_pool.py','render_egad_floating_three.py',
                 'audit_boya_closure_refinement.py','run_boya_floating_diagnostic.py'):
        p=B/name
        sources[str(p)]=sha(p)
    for c in catalog['candidates']:
        g=read(c['generation_json'])
        assert g['complete'] and g['records'][0]['geometry']['ok']
        assert sha(Path(c['candidate_dir'])/'candidates.npz')==g['candidates_sha256']
        inputs.update(g['input_sha256'])
        for name in ('generation.json','candidates.npz','world_object_pose.json'):
            p=Path(c['candidate_dir'])/name
            inputs[str(p)]=sha(p)
        for h in c['eligible_history']:
            p=Path(h['job_path']);j=read(p)
            inputs[str(p)]=sha(p)
            for name in ('evaluation.json','commands.npz','trace.npz','native_audit.json','initialization_audit.json'):
                p=Path(h['output_dir'])/name
                if p.exists():inputs[str(p)]=sha(p)
            inputs[j['protocol_path']]=j['protocol_sha256']
    scores=score_catalog(catalog)
    save(out/'SCORES_PREPHYSICS.json',scores)
    for key in ('input_sha256','source_sha256','model_sha256'):
        if key in scores:
            inputs.update(scores[key])
    derived={str(out/name):sha(out/name) for name in ('CATALOG.json','SCORES_PREPHYSICS.json')}
    # Ranks and all cases are fixed before dispatch; old outcomes remain disclosed.
    queue=planned_queue(catalog['candidates'])
    assert len(queue)==42 and len(set(queue))==42
    cases=[]
    pending={}
    for c in catalog['candidates']:
        cid=c['candidate_id']
        h=c['selected_history']
        case=dict(candidate_id=cid,object_id=c['object_id'],candidate_dir=c['candidate_dir'],
            stable_order_key=c['stable_order_key'],kind='reused' if h else 'new',
            historical_evaluation=h['evaluation_path'] if h else None,
            job_path=h['job_path'] if h else str(out/'jobs'/(cid+'.json')))
        cases.append(case)
        if h:
            continue
        original=ROOT/'outputs/egad_floating_three_20261108_v1/jobs'/(c['object_id']+'.json')
        inputs[str(original)]=sha(original)
        job=copy.deepcopy(read(original))
        job.update(object_id=cid,source_object_id=c['object_id'],candidate_id=cid,
            candidate_dir=c['candidate_dir'],candidate_index=0,candidate_record=c['candidate_record'],
            candidate_rank=c['candidate_record']['global_score_rank'],old_execution=None,
            output_dir=str(out/'execution'/cid),selection_reason='Every still-unexecuted pose in the frozen union; no outcome-dependent selection',
            fixed_pool_catalog=str(out/'CATALOG.json'))
        pending[cid]=job
    check_hashes((sources,inputs,derived))
    protocol=dict(schema='boya_fixed_pool_physical_protocol_v1',
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), cases=cases, new_queue=queue,
        source_sha256=sources,input_sha256=inputs,derived_sha256=derived,
        candidate_count=57,object_candidate_counts=dict(zip(OBJECTS,(31,11,15))),
        reused_candidate_count=15,new_attempts=42,maximum_new_physics_steps=88200,
        new_attempts_per_candidate=1,concurrency=4,maximum_steps_per_attempt=2100,
        stop_on_first_success=False,outcome_dependent_retry=False,training_writeback=False,
        new_CGR_forwards=0,new_training_updates=0,models_changed=False,default_promotion=False,
        no_full_arm=True,controlled_wrist_executor=str(B/'run_boya_floating_diagnostic.py'),
        source_hand_and_collision_geometry_unchanged=True,original_initialization_and_physics_unchanged=True,
        phases_steps=dict(prehold=800,approach=500,close=250,hold=50,lift=500),dt_s=.004,
        score_comparison='Frozen original MSE, preactivation and raw CGR on the SAME finite union; no shared-score model',
        success_rule='Unchanged seven gates, final source-bottom>=5mm, final reference rise>=50mm, final bilateral loads>1mN',
        failure_rule='Original physical early stops count as failed task attempts; infrastructure errors remain unknown and halt new waves',
        historical_reuse='Earliest exact physical-array/configuration-matched controlled-wrist trial; all duplicates disclosed, no outcome-based choice',
        positive_control='Historical A0 coarse successes are separately referenced; not included in the three-object pool denominator',
        purpose='Measure successful-pose coverage and ranking selection within an exhausted finite saved pool',
        limitations=['Known failure-targeted objects; no unseen generalization or deployment claim',
            'Union was previously shortlisted by two fixed scorers; cannot infer absence of a good grasp in the complete CGR space',
            'No complete arm; coarse collision proxies and fixed closure remain potential confounders',
            'One outcome per physical pose plus disclosed historical repeats; not a stochastic grasp-success probability',
            'Oracle-any is hindsight coverage of this finite pool under this executor, never an executable-policy success rate'],
        preparation_wall_s=time.monotonic()-start)
    save(out/'protocol.json',protocol)
    jobs={}
    for cid,job in pending.items():
        job.update(protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'))
        p=out/'jobs'/(cid+'.json')
        save(p,job);jobs[str(p)]=sha(p)
    save(out/'PREPARED.json',dict(protocol_sha256=sha(out/'protocol.json'),jobs_sha256=jobs,
        new_attempts=42,reused_candidates=15,physics_steps=0,rankings_frozen_before_new_physics=True))
    return protocol


def verify(out):
    p=read(out/'protocol.json');ready=read(out/'PREPARED.json')
    assert ready['protocol_sha256']==sha(out/'protocol.json')
    check_hashes((p['source_sha256'],p['input_sha256'],p['derived_sha256'],ready['jobs_sha256']))
    return p


def outcome_row(c,job_path,evaluation,kind):
    e=read(evaluation)
    if not e['complete'] or not 0<e['total_physics_steps']<=2100:
        raise ValueError('Completed physical attempt required: '+str(evaluation))
    return dict(object_id=c['candidate_id'],candidate_id=c['candidate_id'],source_object_id=c['object_id'],
        kind=kind,job_path=str(job_path),job_sha256=sha(job_path),evaluation=str(evaluation),
        evaluation_sha256=sha(evaluation),task_success=e['task_success'],physical_steps=e['total_physics_steps'],
        failure_reason=e['failure_reason'],gates=e['gates'],recorded_execution_object_id=e['object_id'])


def metric_rows(catalog,results):
    out=[]
    for c in catalog['candidates']:
        r=results.get(c['candidate_id'])
        out.append(dict(candidate_id=c['candidate_id'],object_id=c['object_id'],
            status='valid' if r else 'pending',task_success=r['task_success'] if r else None,
            physical_steps=r['physical_steps'] if r else 0,
            physical_gates_passed=all(r['gates'].values()) if r else None,
            early_stop=bool(r['failure_reason']) if r else None))
    return out


def execute(out):
    p=verify(out)
    if (out/'DISPATCH.json').exists() or (out/'RESULTS.json').exists():
        raise ValueError('Already dispatched; never implicitly replay a physical attempt')
    catalog=read(out/'CATALOG.json')
    by_id={c['candidate_id']:c for c in catalog['candidates']}
    results={}
    for c in catalog['candidates']:
        if c['selected_history']:
            h=c['selected_history']
            results[c['candidate_id']]=outcome_row(c,h['job_path'],h['evaluation_path'],'reused')
    save(out/'DISPATCH.json',dict(protocol_sha256=sha(out/'protocol.json'),scores_sha256=sha(out/'SCORES_PREPHYSICS.json'),
        started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pid=os.getpid(),queue=p['new_queue'],concurrency=4))
    began=time.monotonic();errors=[];attempts=[]

    def one(slot,cid):
        job=out/'jobs'/(cid+'.json')
        argv=['taskset','-c',f'{32+8*slot}-{39+8*slot}',sys.executable,'-B',str(B/'run_boya_floating_diagnostic.py'),'--job',str(job)]
        env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
        start=time.monotonic();log=out/'logs'/(cid+'.log')
        with log.open('xb') as f:
            process=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        attempt=dict(candidate_id=cid,argv=argv,returncode=process.returncode,wall_s=time.monotonic()-start,log=str(log))
        save(out/'attempts'/(cid+'.json'),attempt)
        if process.returncode:
            raise RuntimeError(f'{cid}: process {process.returncode}; original log retained')
        evaluation=out/'execution'/cid/'evaluation.json'
        return outcome_row(by_id[cid],job,evaluation,'new'),attempt

    (out/'logs').mkdir()
    for first in range(0,len(p['new_queue']),4):
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures={workers.submit(one,i,cid):cid for i,cid in enumerate(p['new_queue'][first:first+4])}
            for f in as_completed(futures):
                cid=futures[f]
                try:
                    row,attempt=f.result();results[cid]=row;attempts.append(attempt)
                except Exception as exc:
                    errors.append(dict(candidate_id=cid,error=repr(exc)))
                progress=dict(status='executing',new_completed=sum(r['kind']=='new' for r in results.values()),
                    reused=15,new_success=sum(r['kind']=='new' and r['task_success'] for r in results.values()),
                    results=list(results.values()),errors=errors)
                save(out/'PROGRESS.json',progress)
                r=results.get(cid)
                print('FIXED_POOL',cid,dict(success=r['task_success'],steps=r['physical_steps']) if r else errors[-1],flush=True)
        if errors:break
    rows=[results[c['candidate_id']] for c in catalog['candidates'] if c['candidate_id'] in results]
    result=dict(schema='boya_fixed_pool_results_v1',status='complete' if not errors and len(rows)==57 else 'infrastructure_error',
        protocol_sha256=sha(out/'protocol.json'),scores_sha256=sha(out/'SCORES_PREPHYSICS.json'),
        results=rows,candidates=metric_rows(catalog,results),errors=errors,attempts=attempts,
        new_actual_attempts=sum(r['kind']=='new' for r in rows),reused_unique_attempts=sum(r['kind']=='reused' for r in rows),
        new_physics_steps=sum(r['physical_steps'] for r in rows if r['kind']=='new'),
        reused_physics_steps=sum(r['physical_steps'] for r in rows if r['kind']=='reused'),
        physical_steps=sum(r['physical_steps'] for r in rows),success=sum(r['task_success'] for r in rows),
        new_physics_cost_may_be_partial_on_infrastructure_error=bool(errors),
        wall_s=time.monotonic()-began,new_training_updates=0,new_CGR_forwards=0,training_writeback=False,
        independent_generalization=False,default_promotion=False)
    verify(out)
    save(out/'RESULTS.json',result);save(out/'PROGRESS.json',result)
    from fixed_pool_ranking import summarize
    save(out/'METRICS.json',summarize(catalog,read(out/'SCORES_PREPHYSICS.json'),dict(candidates=result['candidates'])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path)
    p.add_argument('--stage',choices=('prepare','execute','verify'),required=True)
    a=p.parse_args();out=a.output.resolve()
    r=prepare(out,a.catalog) if a.stage=='prepare' else globals()[a.stage](out)
    print(json.dumps({k:r[k] for k in ('status','candidate_count','new_attempts','new_actual_attempts','new_physics_steps','success') if k in r}),flush=True)


if __name__=='__main__':
    main()
