"""Plan frozen augmented Boya candidates with the existing four-candidate budget.

Consumes the real comparison schema directly. No compatibility dataset view,
head inference, physics, outcome-based selection or change to the old planner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path
import time
import traceback

from prepare_s40_port_top1_v2 import (Inputs, IDENTITY, rank_candidates,
    prepare_object, verify_selected, read, save, sha)
from planning_recovery import POLICY, prepare_one

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'reproduction/boya'
RECIPES = ('original_mse', 'preactivation')


def catalog(comparison, native_preflight, inputs, recipes=RECIPES):
    comparison = Path(comparison).resolve()
    report = inputs.read(comparison/'RESULTS.json')
    if (report.get('status') != 'complete' or report.get('ranking_labels_read') is not False
            or report.get('new_physics_steps') != 0 or report.get('arm_checked') != 0
            or report.get('frozen_inputs_unchanged') is not True):
        raise ValueError('Complete frozen, label-blind offline comparison required')
    for p, h in report['input_sha256'].items():
        inputs.freeze(p, h)
    native = inputs.read(native_preflight)
    if native.get('complete') is not True or native.get('new_physics_steps') != 0:
        raise ValueError('Complete static zero-step native preflight required')
    out = []
    for recipe in recipes:
        summary = next(r for r in report['cases'] if r['input_kind'] == 'augmented11' and r['recipe'] == recipe)
        case = comparison/('augmented11__'+recipe)
        freeze = inputs.read(case/'RANKING_FREEZE.json')
        if freeze['geometry_calls_before_freeze'] != 0 or freeze['labels_used_for_ranking'] is not False:
            raise ValueError('Ranking must precede full-cloud geometry')
        shortlist = rank_candidates(inputs.read(case/'shortlist.json')['candidates'])
        byrank = {r['global_score_rank']: r for r in shortlist}
        geometry = inputs.read(case/'geometry.json')
        checked = {r['global_score_rank']: r for r in geometry['checked']}
        admitted = geometry['admitted']
        if len(admitted) != summary['geometry_admitted'] or len(checked) != summary['geometry_checked']:
            raise ValueError('Geometry counts differ from comparison')
        candidates = []
        oid = None
        for entry in admitted:
            row = entry['candidate']; rank = entry['global_score_rank']
            if row != checked.get(rank) or row.get('geometry', {}).get('ok') is not True:
                raise ValueError('Admission not bound to checked row')
            if any(row.get(k) != byrank[rank].get(k) for k in IDENTITY):
                raise ValueError('Admission differs from frozen ranking')
            pool = Path(entry['candidate_dir']).resolve()
            if pool.parent != case or pool.name != f'pool_{rank:05d}':
                raise ValueError('Foreign ranked pool')
            generation = inputs.read(pool/'generation.json')
            if generation['records'] != [row] or generation['physics_steps'] != 0 or not generation['complete']:
                raise ValueError('Unbound pool')
            for p, h in generation['input_sha256'].items():
                inputs.freeze(p, h)
            inputs.freeze(pool/'candidates.npz', generation['candidates_sha256'])
            inputs.freeze(pool/'world_object_pose.json')
            obs = inputs.read(generation['source_observation'])
            if oid is not None and oid != obs['object_id']:
                raise ValueError('Mixed object identities')
            oid = obs['object_id']
            source = Path(generation['source_frame'])/'raw_camera_candidates.npz'
            calibration = generation['calibration'] if 'calibration' in generation else str(ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json')
            job = dict(trial_id=f'{oid}_{recipe}_rank_{rank:05d}', object_id=oid,
                split='external_deployment_benchmark', gesture_id=row['gesture_id'],
                extra_depth_index=row['extra_depth_index'], candidate_dir=str(pool),
                observation_json=generation['source_observation'], source_npz=str(source),
                calibration=calibration, candidate_indices=[0], max_arm_plans=1)
            candidates.append(dict(row=row, job=job))
        if oid is None:
            # Protocol is an explicit input even when no candidate survives.
            protocol = inputs.read(next(p for p, h in report['input_sha256'].items()
                if Path(p).name == 'COMPARISON_PROTOCOL.json'))
            oid = protocol['object_id']
        n = next(r for r in native['results'] if r['object_id'] == oid)
        for p in n.get('artifacts', []):
            inputs.freeze(p)
        if n.get('result_json'):
            inputs.freeze(n['result_json'])
        if n.get('repair_path'):
            inputs.freeze(n['repair_path'], n['repair_sha256'])
            for candidate in candidates:
                candidate['job'].update(collision_refinement=n['repair_path'], collision_refinement_sha256=n['repair_sha256'])
        ordered = rank_candidates([v['row'] for v in candidates])
        lookup = {v['row']['global_score_rank']:v for v in candidates}
        out.append(dict(object_id=oid, recipe=recipe,
            candidates=[lookup[v['global_score_rank']] for v in ordered[:4]],
            geometry_admitted_total=len(candidates), comparison_dir=str(comparison)))
    return out


def run(comparisons, native_preflight, output, backend, workers=3):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing planning output is never overwritten')
    inputs = Inputs(); inputs.freeze(backend)
    items = [v for p in comparisons for v in catalog(p, native_preflight, inputs)]
    keys = [(v['object_id'],v['recipe']) for v in items]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate object/recipe')
    sources = {str(p):sha(p) for p in [Path(__file__).resolve(), B/'planning_recovery.py',
        B/'prepare_s40_port_top1_v2.py',B/'prepare_collection_queue.py', B/'native_arm.py', B/'trial_record.py']}
    inputs.verify()
    output.mkdir(parents=True)
    save(output/'PLAN.json',dict(schema='boya_augmented_full_arm_planning_v1',
        policy=POLICY, arm_candidates_per_object_recipe=4, physical_steps=0,
        jobs=items, source_sha256=sources, input_sha256=inputs.files,
        physical_attempts_authorized_by_this_entry=0))
    began=time.monotonic(); results={}; errors=[]
    def one(item):
        directory=output/item['recipe']; directory.mkdir(exist_ok=True)
        result=prepare_object(copy.deepcopy(item),directory,Path(backend),prepare_one,verify_selected)
        result['recipe']=item['recipe']
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(one,item):item for item in items}
        for future in as_completed(jobs):
            item=jobs[future]; key=(item['object_id'],item['recipe'])
            try:results[key]=future.result()
            except Exception:errors.append(dict(object_id=key[0],recipe=key[1],error=traceback.format_exc()))
            progress=dict(status='planning',completed=len(results),errors=errors,total=len(items))
            from prepare_egad_external_scenes import save as replace
            replace(output/'PROGRESS.json',progress)
            print('ARM_PLANNED',key,results.get(key,{}).get('status','error'),flush=True)
    inputs.verify()
    if any(sha(p)!=h for p,h in sources.items()):raise ValueError('Planner source changed')
    report=dict(schema='boya_augmented_full_arm_planning_results_v1',status='complete' if not errors else 'infrastructure_error',
        results=[results[k] for k in keys if k in results],errors=errors,
        physical_steps=0,training_writeback=False,arm_budget_per_object_recipe=4,
        input_sha256=inputs.files,source_sha256=sources,inputs_unchanged=True,wall_s=time.monotonic()-began)
    save(output/'RESULTS.json',report)
    if errors:raise RuntimeError('Planning incomplete; all errors retained')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,nargs='+',required=True)
    parser.add_argument('--native-preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',type=Path,default=ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json')
    parser.add_argument('--workers',type=int,choices=range(1,5),default=3)
    args=parser.parse_args();run(args.comparison,args.native_preflight,args.output,args.backend,args.workers)
