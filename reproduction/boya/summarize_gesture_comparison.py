#!/usr/bin/env python3
"""Read-only Boya 9x8 gesture comparison; no simulator, NumPy, model or learner.

The old three gesture records are inherited, never replayed. See the adjacent
GESTURE_COMPARISON_20261105.md for denominators and the post-hoc subset boundary.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == 'boya' else Path('/home/wangrenpeng/AnyDexGrasp')
OLD = (1, 6, 8)
NEW = (2, 3, 5, 7, 9, 10)
GESTURES = (1, 2, 3, 5, 6, 7, 8, 9, 10)
NAMES = {1:'五指握',2:'四指握',3:'三指握',5:'五指扣握',6:'二指捏合',7:'三指捏合',8:'爪式',9:'夹式',10:'钳式'}
TERMINAL = {'complete','complete_with_null','complete_with_infrastructure_errors','no_candidate','no_hand_geometry_in_declared_budget','no_remaining_candidates','no_full_arm_plan_in_declared_budget','execution_infrastructure_error','infrastructure_error','cancelled','frozen_pool_exhausted','actual_execution_null'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self): self.inputs = {}
    def freeze(self, path, expected=None):
        path = str(Path(path).resolve()); actual = sha(path)
        if expected is not None and actual != expected: raise ValueError('SHA mismatch: '+path)
        if path in self.inputs and self.inputs[path] != actual: raise ValueError('Input changed: '+path)
        self.inputs[path] = actual
        return actual
    def read(self, path):
        before = self.freeze(path); value = json.loads(Path(path).read_text())
        self.freeze(path, before)
        return value
    def expected(self, files):
        for path, expected in files.items(): self.freeze(path, expected)
    def unchanged(self):
        return all(Path(p).is_file() and sha(p) == h for p,h in self.inputs.items())


def valid_label(evaluation):
    """The existing bound execution contract; zero/infra/null never become 0."""
    steps = evaluation.get('total_physics_steps')
    if type(steps) is not int or steps <= 0: return None
    if any(evaluation.get(k) for k in ('infrastructure_error','error','exception','errors')): return None
    if evaluation.get('complete') is not True or evaluation.get('label_eligible') is not True: return None
    if 'lift' not in evaluation.get('completed_phases', []): return None
    success = evaluation.get('task_success')
    return int(success) if type(success) is bool else None


def resolve_round(reader, item, seen_records, expected_object, expected_gesture, expected_source, expected_observation):
    row = {'directory': item} if isinstance(item, str) else dict(item)
    directory = Path(row['directory']).resolve()
    record_directory = Path(row.get('_record_directory',directory)).resolve()
    summary_path = directory/'trial_summary.json'
    if not summary_path.is_file(): summary_path = directory/'preparation.json'
    summary = reader.read(summary_path) if summary_path.is_file() else {}
    execution = row.get('execution') or {}
    native_path = Path(execution['execution_dir'])/'evaluation.json' if execution.get('execution_dir') else directory/'execution/evaluation.json'
    plans = summary.get('plans', row.get('plans', []))
    status = summary.get('status', row.get('status','pending'))
    steps = max(row.get('physics_steps',row.get('actual_physics_steps',0)) or 0,summary.get('physics_steps',0) or 0,execution.get('physics_steps',0) or 0)
    if type(steps) is not int or steps < 0: raise ValueError('Malformed steps: '+str(directory))
    record_path = record_directory/'training_record.json'
    label = None; binding = {}; artifacts = {}; eligible = False; reason = status
    if record_path.is_file():
        if str(record_path) in seen_records: raise ValueError('Repeated physical record: '+str(record_path))
        seen_records.add(str(record_path)); record = reader.read(record_path)
        binding = record['binding']; bd = binding['trial_binding_sha256']
        if digest({k:v for k,v in binding.items() if k != 'trial_binding_sha256'}) != bd: raise ValueError('Edited binding')
        if binding['multifinger_pose_finger_type'] != expected_gesture or binding['multifinger_pose_depth_type'] != 0: raise ValueError('Wrong gesture/extra depth in record')
        if binding['source_npz_sha256'] != reader.freeze(expected_source): raise ValueError('Different source CGR')
        reader.freeze(binding['source_npz'],binding['source_npz_sha256'])
        if binding['execution_spec'].get('source_observation_sha256') != reader.freeze(expected_observation): raise ValueError('Different observation in record')
        artifacts = record['execution_artifacts']
        evaluation = reader.read(artifacts['evaluation_json'])
        reader.freeze(artifacts['evaluation_json'],artifacts['evaluation_sha256'])
        command = Path(artifacts['commands_npz']);reader.freeze(command,artifacts['commands_sha256'])
        if evaluation.get('trial_binding_sha256') != bd or evaluation.get('commands_sha256') != artifacts['commands_sha256']: raise ValueError('Actual commands not bound')
        plan_path = command.parent/'plan.json'
        plan = reader.read(plan_path)
        if plan['gesture_id'] != expected_gesture or Path(plan['arguments']['object_root']).name != expected_object: raise ValueError('Plan object/gesture mismatch')
        if plan.get('commands_sha256') != artifacts['commands_sha256'] or plan.get('trial_binding_sha256') != bd: raise ValueError('Plan/commands binding mismatch')
        native = native_path
        if not native.is_file(): raise ValueError('Missing actual native evaluation')
        ev = reader.read(native)
        if ev.get('plan_sha256'): reader.freeze(plan_path,ev['plan_sha256'])
        if evaluation.get('original_native_evaluation_sha256') != reader.freeze(native): raise ValueError('Native evaluation not bound')
        reader.freeze(native.parent/'trace.npz',ev['trace_sha256'])
        if ev.get('commands_sha256') != artifacts['commands_sha256'] or ev.get('total_physics_steps') != evaluation.get('total_physics_steps'): raise ValueError('Native execution differs')
        for key in ('complete','task_success'):
            if ev.get(key) != evaluation.get(key): raise ValueError('Bound evaluation changes native outcome')
        label = valid_label(evaluation)
        if label is not None and evaluation['total_physics_steps'] != 2100: raise ValueError('Valid label without original full 2100 steps')
        if record.get('result') != label or record.get('label_eligible') is not (label is not None): raise ValueError('Record labels violate execution contract')
        if label is not None and type(record['result']) is not int: raise ValueError('Non-integer training label')
        if evaluation.get('object_root') and Path(evaluation['object_root']).name != expected_object: raise ValueError('Executed wrong object')
        steps = evaluation['total_physics_steps'];eligible = label is not None; reason = record.get('outcome_status',status)
        eligible = label is not None
        if row.get('training_record_sha256'): reader.freeze(record_path,row['training_record_sha256'])
        if row.get('binding_json'):
            source_binding = reader.read(row['binding_json'])
            if row.get('binding_sha256'): reader.freeze(row['binding_json'],row['binding_sha256'])
            if source_binding != binding: raise ValueError('Prepared source binding differs from actual record')
        if row.get('commands_sha256') and row['commands_sha256'] != artifacts['commands_sha256']: raise ValueError('Prepared commands differ from actual commands')
        if execution:
            if execution.get('trial_id') != row.get('trial_id'): raise ValueError('Wrong actual trial ID')
            if execution.get('commands_sha256') != artifacts['commands_sha256'] or execution.get('trial_binding_sha256') != bd: raise ValueError('Actual queue identity differs')
            if Path(execution['source_plan_dir']).resolve() != Path(row['plan_dir']).resolve(): raise ValueError('Actual queue executes a different prepared plan')
            if execution.get('label') != label or execution.get('physics_steps') != steps or execution.get('label_eligible') is not eligible: raise ValueError('Queue label/steps disagree with actual record')
            guards=execution.get('guards',{})
            if guards != evaluation.get('queue_binding_guards',{}): raise ValueError('Queue guards differ from bound evaluation')
            if label is not None and (len(guards)!=11 or not all(v is True for v in guards.values())): raise ValueError('Valid label without all eleven queue guards')
        if row.get('training_label') is not None and row['training_label'] != label: raise ValueError('Ledger label differs')
    else:
        native = native_path
        if native.is_file():
            ev = reader.read(native);steps = max(steps,ev.get('total_physics_steps',0) or 0)
            if (native.parent/'trace.npz').is_file() and ev.get('trace_sha256'): reader.freeze(native.parent/'trace.npz',ev['trace_sha256'])
    if steps > 2100: raise ValueError('Unexpected execution length')
    if execution.get('label') is not None and not record_path.is_file(): raise ValueError('Queue claims a label without an actual bound record')
    if record_path.is_file(): status='complete' if label is not None else 'actual_execution_null'
    elif row.get('_runtime_round') and row.get('status')=='ready': status='not_dispatched' if not execution else status
    selected = summary.get('selected_candidate',row.get('selected_candidate'))
    admitted = [p for p in plans if p.get('admitted') is True]
    return dict(directory=str(directory),trial_id=row.get('trial_id'),status=status,terminal=(row.get('terminal',status in TERMINAL) and not (row.get('_runtime_round') and row.get('status')=='ready' and not execution)),physics_steps=steps,actual=steps>0,label=label,label_eligible=eligible,null_reason=None if eligible else reason,unconfirmed_execution_attempts=row.get('unconfirmed_execution_attempts',0),arm_attempts=len(plans),arm_admitted=len(admitted),arm_rejected=sum(p.get('admitted') is False for p in plans),arm_candidate_indices=[p.get('index') for p in plans],executed_pool_index=admitted[-1].get('index') if steps and admitted else None,source_CGR_row=binding.get('source_row_index'),if_flip=binding.get('if_flip'),binding_sha256=binding.get('trial_binding_sha256'),record_json=str(record_path) if record_path.is_file() else None,record_sha256=reader.freeze(record_path) if record_path.is_file() else None,selected_candidate=selected)


def geometry(reader, cell, expected_source, expected_observation, gesture):
    if not cell.get('pool'): return dict(status='pending',covered=None,kept_candidates=None,checks=[])
    pool = Path(cell['pool']);gp = pool/'generation.json';np = pool/'no_candidate.json'
    if not gp.is_file() and not np.is_file(): return dict(status='pending',covered=None,kept_candidates=None,pool=str(pool),checks=[])
    path = gp if gp.is_file() else np;g = reader.read(path)
    if int(g.get('gesture_id',gesture)) != gesture: raise ValueError('Wrong pool gesture')
    if g.get('source_observation') and reader.freeze(g['source_observation']) != reader.freeze(expected_observation): raise ValueError('Pool observation mismatch')
    if g.get('source_frame') and reader.freeze(Path(g['source_frame'])/'raw_camera_candidates.npz') != reader.freeze(expected_source): raise ValueError('Pool CGR mismatch')
    reader.expected(g.get('input_sha256',{}))
    if g.get('source_candidates_sha256') and g['source_candidates_sha256'] != reader.freeze(expected_source): raise ValueError('Null pool CGR mismatch')
    if g.get('source_observation_sha256') and g['source_observation_sha256'] != reader.freeze(expected_observation): raise ValueError('Null pool observation mismatch')
    records = g.get('records',[])
    if any(r.get('additional_depth_index',0) != 0 or r.get('additional_depth_m',0) != 0 for r in records): raise ValueError('Pool changed extra depth')
    if any(r.get('jaw_assignment',{}).get('opponent','ff') != 'ff' or (r.get('source_slot') and not r['source_slot'].endswith(':ff')) for r in records): raise ValueError('Pool changed ff datum')
    indices = g.get('selected_indices',list(range(len(records)))) if gp.is_file() else []
    if len(indices) > 8: raise ValueError('Pool keep exceeds eight')
    if gp.is_file():
        reader.freeze(pool/'candidates.npz',g['candidates_sha256'])
        if len(indices) == 0: raise ValueError('Ready generation without candidates')
    count = len(indices) if gp.is_file() else 0
    return dict(status='ready' if count else 'no_candidate',covered=bool(count),kept_candidates=count,pool=str(pool),receipt=str(path),receipt_sha256=reader.freeze(path),checks=['source hashes if published','extra_depth0','ff','keep<=8'],declared_candidate_budget=500)


def summarize_cell(reader, cell, old_object, seen_records, origin):
    oid = cell['object_id'];gid = int(cell['gesture_id']);source = Path(old_object['representation_dir'])/'raw_camera_candidates.npz';observation = old_object.get('observation',old_object.get('observation_json'))
    reader.freeze(source);reader.freeze(observation)
    dynamic = {}
    if cell.get('result_json') and Path(cell['result_json']).is_file(): dynamic = reader.read(cell['result_json'])
    for k in ('object_id','gesture_id'):
        if k in dynamic and dynamic[k] != cell[k]: raise ValueError('Cell receipt identity mismatch')
    current = dict(cell,**dynamic);items = current.get('rounds',current.get('round_dirs',[]))
    rounds = [resolve_round(reader,r,seen_records,oid,gid,source,observation) for r in items]
    if len(rounds)>2: raise ValueError('More than two rounds')
    if sum(r['arm_attempts'] for r in rounds)>8: raise ValueError('More than eight arm candidates')
    if any(r['arm_attempts']>4 for r in rounds): raise ValueError('More than four arm candidates per round')
    consumed = [i for r in rounds for i in r['arm_candidate_indices']]
    if len(consumed) != len(set(consumed)): raise ValueError('Repeated arm candidate within a cell')
    actual = [r for r in rounds if r['actual']];labels = [r['label'] for r in actual if r['label'] is not None]
    if len(actual)>2: raise ValueError('More than two actual executions')
    status = current.get('status','pending');terminal = status in TERMINAL or current.get('terminal') is True
    if current.get('_runtime_cell'):
        terminal = len(rounds)==2 and all(r['terminal'] for r in rounds)
        if terminal: status='complete' if any(r['actual'] for r in rounds) else 'complete_without_actual'
    geom = geometry(reader,current,source,observation,gid)
    if geom['covered'] is False and not rounds and status=='pending': status='no_candidate';terminal=True
    success = any(r['label']==1 for r in actual)
    first_success = next((i+1 for i,r in enumerate(actual) if r['label']==1),None)
    rank0 = [r for r in rounds if 0 in r['arm_candidate_indices']]
    rank0_actual = next((r for r in actual if r['executed_pool_index']==0),None)
    unexecuted_slots = max(0,2-len(rounds)) if terminal else 0
    null_slots = sum(r['label'] is None and (r['actual'] or r['terminal']) for r in rounds)+unexecuted_slots
    return dict(object_id=oid,gesture_id=gid,gesture_name=NAMES[gid],origin=origin,status=status,terminal=terminal,geometry=geom,arm_covered=True if any(r['arm_admitted'] for r in rounds) else False if terminal else None,arm_attempts=sum(r['arm_attempts'] for r in rounds),arm_admitted=sum(r['arm_admitted'] for r in rounds),arm_rejected=sum(r['arm_rejected'] for r in rounds),actual_attempts=len(actual),positive_labels=labels.count(1),negative_labels=labels.count(0),valid_actual_labels=len(labels),actual_nulls=sum(r['label'] is None for r in actual),zero_step_rounds=sum(not r['actual'] for r in rounds),unconfirmed_execution_attempts=sum(r['unconfirmed_execution_attempts'] for r in rounds),physics_steps=sum(r['physics_steps'] for r in rounds),first_actual_success=actual[0]['label'] if actual else None,first_actual_valid=bool(actual and actual[0]['label'] is not None),observed_success_within_max2=success,success_coverage_final=bool(success) if terminal or success else None,attempts_to_first_success=first_success,two_valid_actuals=len(actual)==2 and len(labels)==2,two_for_two_success=bool(len(actual)==2 and len(labels)==2 and labels==[1,1]),planned_maximum_slots=2,terminal_null_slots=null_slots,terminal_unexecuted_slots=unexecuted_slots,round1_outcome=rounds[0]['label'] if rounds else None,round1_status=rounds[0]['status'] if rounds else status,historical_execution_nulls=sum(r['label'] is None and ('infrastructure' in r['status'] or r['unconfirmed_execution_attempts']>0) for r in rounds),rank0=dict(arm_attempted=bool(rank0),actually_executed=rank0_actual is not None,label=rank0_actual['label'] if rank0_actual else None),rounds=rounds)


def aggregate(cells):
    den=sum(c['valid_actual_labels'] for c in cells);first=[c['first_actual_success'] for c in cells if c['first_actual_valid']];both=[c for c in cells if c['two_valid_actuals']]
    return dict(fixed_cells=len(cells),terminal_cells=sum(c['terminal'] for c in cells),pending_cells=sum(not c['terminal'] for c in cells),geometry_covered=sum(c['geometry']['covered'] is True for c in cells),geometry_known=sum(c['geometry']['covered'] is not None for c in cells),arm_covered=sum(c['arm_covered'] is True for c in cells),arm_coverage_known=sum(c['arm_covered'] is not None for c in cells),first_actual_successes=sum(first),first_actual_valid_denominator=len(first),first_actual_rate=sum(first)/len(first) if first else None,observed_max2_success_cells=sum(c['observed_success_within_max2'] for c in cells),actual_attempts=sum(c['actual_attempts'] for c in cells),valid_actual_labels=den,positive_labels=sum(c['positive_labels'] for c in cells),negative_labels=sum(c['negative_labels'] for c in cells),actual_success_rate=sum(c['positive_labels'] for c in cells)/den if den else None,actual_nulls=sum(c['actual_nulls'] for c in cells),cells_without_actual=sum(c['actual_attempts']==0 for c in cells),zero_step_rounds=sum(c['zero_step_rounds'] for c in cells),unconfirmed_execution_attempts=sum(c['unconfirmed_execution_attempts'] for c in cells),planned_maximum_slots=2*len(cells),terminal_null_slots=sum(c['terminal_null_slots'] for c in cells),terminal_unexecuted_slots=sum(c['terminal_unexecuted_slots'] for c in cells),historical_execution_nulls=sum(c['historical_execution_nulls'] for c in cells),round1_positive_labels=sum(c['round1_outcome']==1 for c in cells),round1_valid_label_denominator=sum(c['round1_outcome'] is not None for c in cells),two_for_two_cells=sum(c['two_for_two_success'] for c in both),two_valid_actuals_denominator=len(both),arm_attempts=sum(c['arm_attempts'] for c in cells),arm_admitted=sum(c['arm_admitted'] for c in cells),physics_steps=sum(c['physics_steps'] for c in cells),rank0_actual_positive=sum(c['rank0']['label']==1 for c in cells),rank0_actual_valid_denominator=sum(c['rank0']['label'] is not None for c in cells))


def comparison(cells,object_ids):
    by={(c['object_id'],c['gesture_id']):c for c in cells};baseline={o for o in object_ids if any(by[o,g]['observed_success_within_max2'] for g in OLD)}
    rows=[]
    for g in NEW:
        covered={o for o in object_ids if by[o,g]['observed_success_within_max2']}
        repeat=[];fewer=[]
        for o in object_ids:
            c=by[o,g]
            bs=[by[o,b] for b in OLD]
            if c['two_for_two_success']:
                worse=[b['gesture_id'] for b in bs if b['two_valid_actuals'] and not b['two_for_two_success']]
                if worse: repeat.append(dict(object_id=o,old_gestures_with_two_valid_but_not_two_success=worse))
            successes=[b for b in bs if b['attempts_to_first_success'] is not None]
            if c['attempts_to_first_success'] is not None and successes:
                best=min(b['attempts_to_first_success'] for b in successes)
                if c['attempts_to_first_success']<best:fewer.append(dict(object_id=o,new_actual_attempts=c['attempts_to_first_success'],old_best_actual_attempts=best,new_arm_attempts=c['arm_attempts'],old_successful_arm_attempt_counts={str(b['gesture_id']):b['arm_attempts'] for b in successes}))
        rows.append(dict(gesture_id=g,observed_success_object_ids=sorted(covered),new_object_ids_over_old_union=sorted(covered-baseline),observed_two_repeat_consistency_improvements=repeat,observed_fewer_real_attempts=fewer))
    newunion={o for o in object_ids if any(by[o,g]['observed_success_within_max2'] for g in NEW)}
    return dict(baseline_gestures=list(OLD),baseline_observed_success_objects=sorted(baseline),new_six_observed_success_objects=sorted(newunion),new_object_ids_over_baseline=sorted(newunion-baseline),by_new_gesture=rows,boundary='Observed development-case differences; two repeats do not identify success probabilities.')


def subset_analysis(cells,object_ids):
    masks={g:{c['object_id'] for c in cells if c['gesture_id']==g and c['observed_success_within_max2']} for g in GESTURES}
    costs={g:sum(c['actual_attempts'] for c in cells if c['gesture_id']==g) for g in GESTURES}
    base=set().union(*(masks[g] for g in OLD));rows=[]
    for bits in range(1<<len(GESTURES)):
        gs=[g for i,g in enumerate(GESTURES) if bits&(1<<i)];cover=set().union(*(masks[g] for g in gs)) if gs else set();cost=sum(costs[g] for g in gs);newgs=[g for g in gs if g not in OLD];increment=set().union(*(masks[g] for g in newgs))-base if newgs else set();newcost=sum(costs[g] for g in newgs)
        rows.append(dict(gestures=gs,size=len(gs),covered_object_ids=sorted(cover),covered_count=len(cover),observed_actual_attempts=cost,covered_objects_per_observed_actual=len(cover)/cost if cost else None,additional_objects_over_baseline=sorted(increment),additional_observed_actual_attempts=newcost,additional_objects_per_added_actual=len(increment)/newcost if newcost else None))
    union=set().union(*masks.values());full=[r for r in rows if r['covered_count']==len(union)];minsize=min(r['size'] for r in full)
    pairs=[]
    for a,b in itertools.combinations(GESTURES,2):
        u=masks[a]|masks[b];pairs.append(dict(gestures=[a,b],jaccard=len(masks[a]&masks[b])/len(u) if u else None,same_observed_success_set=masks[a]==masks[b]))
    return dict(scope='Post-hoc observed-case coverage upper bound, not learned or deployable policy success.',partial=any(not c['terminal'] for c in cells),subset_count=len(rows),observed_union_objects=sorted(union),minimum_cardinality_to_cover_observed_union=minsize,minimum_cover_subsets=[r['gestures'] for r in full if r['size']==minsize],best_coverage_by_size=[dict(size=k,covered_count=max(r['covered_count'] for r in rows if r['size']==k)) for k in range(10)],leave_one_out_coverage_loss={str(g):sorted(union-set().union(*(masks[h] for h in GESTURES if h!=g))) for g in GESTURES},pairwise_redundancy=pairs,subsets=rows)


def normalize_cells(payload):
    if isinstance(payload,list):return payload
    cells=payload.get('cells',payload.get('jobs',[]))
    if isinstance(cells,dict):return list(cells.values())
    return cells


def markdown(report):
    by={(c['object_id'],c['gesture_id']):c for c in report['cells']}
    lines=['# Boya九手型开发集比较','',report['scope'],'','单元格：正标签/有效实际标签；P=尚未完成，N=终态无实际，U=实际执行但无有效标签。','', '| 物体 | '+' | '.join(f'{g} {NAMES[g]}' for g in GESTURES)+' |','|---|'+'---|'*len(GESTURES)]
    for o in report['object_ids']:
        vals=[]
        for g in GESTURES:
            c=by[o,g];v=f"{c['positive_labels']}/{c['valid_actual_labels']}"
            if not c['terminal']:v+=' P'
            elif not c['actual_attempts']:v+=' N'
            if c['actual_nulls']:v+=' U'
            vals.append(v)
        lines.append('| '+o+' | '+' | '.join(vals)+' |')
    lines+=['','首次实际成功的分母只含有效首次实际标签；无候选/臂拒绝/null不补负标签。','', '| 手型 | 几何覆盖/8 | 臂覆盖/8 | 首实际正/有效首实际 | ≤2次已观察成功物体/8 | 正/有效实际 | 实际null | 两次均正/两次有效 |','|---|---|---|---|---|---|---|---|']
    for row in report['per_gesture']:
        x=row['metrics'];lines.append(f"| {row['gesture_id']} {row['name']} | {x['geometry_covered']}/8 | {x['arm_covered']}/8 | {x['first_actual_successes']}/{x['first_actual_valid_denominator']} | {x['observed_max2_success_cells']}/8 | {x['positive_labels']}/{x['valid_actual_labels']} | {x['actual_nulls']} | {x['two_for_two_cells']}/{x['two_valid_actuals_denominator']} |")
    s=report['subset_analysis'];lines+=['',f"新增六型相对旧三型的已观察新增物体：{report['relative_to_old_three']['new_object_ids_over_baseline']}。",'',f"512子集事后覆盖分析：当前已观察并集{s['observed_union_objects']}；覆盖该并集的最小基数{s['minimum_cardinality_to_cover_observed_union']}。partial={s['partial']}。",'','这只是已观察开发案例的事后覆盖上界，不是learned或在线选型策略的成功率；详细成本、冗余和全部512子集见RESULTS.json。','']
    return '\n'.join(lines)


def runtime_payload(reader, source):
    """Native run_gesture_comparison_trials.py files, without virtual legacy dirs."""
    source=Path(source).resolve()
    if source.is_dir():
        root=source;path=root/'RESULTS.json' if (root/'RESULTS.json').is_file() else root/'PREPARED.json'
    else:
        path=source;root=path.parent
    payload=reader.read(path)
    if payload.get('schema')!='anydex_boya_gesture_comparison_v1' and not (root/'cell_inputs').is_dir():
        return payload,dict(source=str(path),runtime=False)
    prepared=reader.read(root/'PREPARED.json')
    reader.expected(prepared['input_sha256']);reader.expected(prepared['source_sha256'])
    cells=[]
    for raw in normalize_cells(payload):
        cid=raw['cell_id'];source_path=root/'cell_inputs'/(cid+'.json')
        frozen=reader.read(source_path);reader.expected(frozen['input_sha256']);reader.expected(frozen['source_sha256'])
        cell=dict(frozen['cell'],**raw);cell['_runtime_cell']=True;rounds=[]
        for item in raw.get('rounds',[]):
            row=dict(item);trial=row['trial_id']
            if Path(trial).name!=trial or trial in ('.','..'): raise ValueError('Unsafe trial identity')
            source_dir=Path(row['binding_json']).parent if row.get('binding_json') else root/'plans'/trial
            row.update(directory=str(source_dir),_record_directory=str(root/'execution'/cell['object_id']/'trials'/trial),_runtime_round=True)
            rounds.append(row)
        cell['rounds']=rounds;cells.append(cell)
    return dict(cells=cells),dict(source=str(path),run_root=str(root),runtime=True,runtime_status=payload.get('status'),runtime_final=path.name=='RESULTS.json',prepared_sha256=reader.freeze(root/'PREPARED.json'),reported_label_count=payload.get('label_count'),reported_positive_labels=payload.get('positive_labels'),reported_negative_labels=payload.get('negative_labels'),reported_actual_physics_steps=payload.get('actual_physics_steps'))


def run(args):
    start=time.monotonic();reader=Reader();oldroot=Path(args.old_pilot);protocol=reader.read(oldroot/'protocol_v3.json');prepared=reader.read(oldroot/'PREPARED.json');progress=reader.read(oldroot/'progress.json')
    reader.expected(prepared['frozen_inputs']);object_ids=protocol['object_ids'];objects={o['id']:o for o in protocol['objects']}
    if len(object_ids)!=8 or set(protocol['gesture_ids'])!=set(OLD):raise ValueError('Expected original eight objects and three inherited gestures')
    old_records=[json.loads(s) for s in (oldroot/'training_records.jsonl').read_text().splitlines()];reader.freeze(oldroot/'training_records.jsonl')
    if len(old_records)!=36:raise ValueError('Expected unchanged old36 valid records')
    oldjobs={j['id']:j for j in prepared['jobs']};cells=[];seen=set();provided={}
    selected_source=args.run_root or args.source or args.cells
    if args.run_root and args.source:raise ValueError('Choose a positional source or --run-root')
    payload,new_runtime=runtime_payload(reader,selected_source) if selected_source else ({'cells':[]},{'runtime':False})
    for c in normalize_cells(payload):
        key=(c['object_id'],int(c['gesture_id']))
        if key in provided or key[0] not in objects or key[1] not in NEW:raise ValueError('Duplicate or foreign new cell')
        provided[key]=c
    if args.new_progress:
        for c in normalize_cells(reader.read(args.new_progress)):
            key=(c['object_id'],int(c['gesture_id']))
            if key[0] not in objects or key[1] not in NEW:raise ValueError('Foreign progress cell')
            provided[key]=dict(provided.get(key,{}),**c)
    for oid in object_ids:
        for gid in GESTURES:
            if gid in OLD:
                jid=f'{oid}__g{gid:02d}';job=oldjobs[jid];state=progress['jobs'][jid];cell=dict(job,**state);origin='inherited_original_pilot_no_rerun'
            else:cell=provided.get((oid,gid),dict(object_id=oid,gesture_id=gid,status='pending'));origin='new_six_gesture_comparison'
            cells.append(summarize_cell(reader,cell,objects[oid],seen,origin))
    inherited=[c for c in cells if c['gesture_id'] in OLD];a=aggregate(inherited)
    if (a['actual_attempts'],a['positive_labels'],a['negative_labels'],a['physics_steps'])!=(36,29,7,75600):raise ValueError('Inherited pilot counts changed')
    inherited_record_paths={r['record_json'] for c in inherited for r in c['rounds'] if r['record_json']}
    if len(inherited_record_paths)!=36 or [reader.read(p) for p in sorted(inherited_record_paths)]!=sorted(old_records,key=lambda r:r['execution_artifacts']['evaluation_json']):
        # Compare dictionaries by bound record identity rather than file order.
        left={reader.read(p)['binding']['trial_binding_sha256']:reader.read(p) for p in inherited_record_paths};right={r['binding']['trial_binding_sha256']:r for r in old_records}
        if left!=right:raise ValueError('Inherited export differs from original records')
    if args.require_complete and new_runtime.get('runtime') and (not new_runtime['runtime_final'] or new_runtime['runtime_status']!='complete'):raise ValueError('Runtime RESULTS is not complete')
    if args.require_complete and (len(provided)!=48 or any(not c['terminal'] for c in cells if c['gesture_id'] in NEW)):raise ValueError('New 48 cells not all complete')
    new_cells=[c for c in cells if c['gesture_id'] in NEW]
    if new_runtime.get('runtime_final'):
        actual_records={r['record_json']:reader.read(r['record_json']) for c in new_cells for r in c['rounds'] if r['record_json'] and r['label'] is not None}
        export_path=Path(new_runtime['run_root'])/'training_records.jsonl'
        reader.freeze(export_path);export=[json.loads(line) for line in export_path.read_text().splitlines()]
        actual_by_binding={rec['binding']['trial_binding_sha256']:rec for rec in actual_records.values()}
        expected_context={r['binding_sha256']:dict(trial_id=r['trial_id'],object_id=c['object_id'],gesture_id=c['gesture_id']) for c in new_cells for r in c['rounds'] if r['record_json'] and r['label'] is not None}
        if len(actual_by_binding)!=len(actual_records):raise ValueError('Duplicate new bindings')
        exported={}
        for rec in export:
            enriched=dict(rec);context=enriched.pop('collection_context',None);bd=enriched['binding']['trial_binding_sha256']
            if not context or bd in exported or context.get('gesture_id') not in NEW or context.get('extra_depth_index')!=0 or context.get('comparison_protocol_sha256')!=new_runtime['prepared_sha256']:raise ValueError('Invalid new export context')
            if any(context.get(k)!=v for k,v in expected_context.get(bd,{}).items()) or bd not in expected_context:raise ValueError('Wrong per-trial export context')
            if enriched!=actual_by_binding.get(bd):raise ValueError('Export differs from actual bound record')
            exported[bd]=enriched
        if exported!=actual_by_binding:raise ValueError('New export omits or adds labels')
        na=aggregate(new_cells)
        for key,expected in [('reported_label_count',na['valid_actual_labels']),('reported_positive_labels',na['positive_labels']),('reported_negative_labels',na['negative_labels']),('reported_actual_physics_steps',na['physics_steps'])]:
            if new_runtime[key]!=expected:raise ValueError('Runtime aggregate differs: '+key)
        new_runtime['independent_export_record_count']=len(export);new_runtime['export_sha256']=reader.freeze(export_path)
    report=dict(schema='anydex_boya_gesture_comparison_summary_v1',scope='Eight known development objects; inherited three gestures and six added gestures. Frozen CGR, zero extra insertion. No unknown-object optimality or learned-selection result.',object_ids=object_ids,gesture_ids=list(GESTURES),inherited_gesture_ids=list(OLD),new_gesture_ids=list(NEW),new_cells_supplied=len(provided),new_runtime=new_runtime,complete=all(c['terminal'] for c in cells),aggregate=aggregate(cells),inherited_aggregate=a,new_aggregate=aggregate([c for c in cells if c['gesture_id'] in NEW]),per_gesture=[dict(gesture_id=g,name=NAMES[g],metrics=aggregate([c for c in cells if c['gesture_id']==g])) for g in GESTURES],cells=cells,relative_to_old_three=comparison(cells,object_ids),subset_analysis=subset_analysis(cells,object_ids),input_sha256=reader.inputs,summary_source_sha256=sha(__file__),predeclared_metrics=str(ROOT/'reproduction/boya/GESTURE_COMPARISON_20261105.md'),cost=dict(new_physics_steps=0,new_network_forwards=0,new_training_steps=0,elapsed_wall_s=time.monotonic()-start))
    if not reader.unchanged():raise ValueError('Source changed while summarizing; use a final or frozen progress snapshot')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n');(out/'SUMMARY.md').write_text(markdown(report))
    print(json.dumps(dict(output=str(out),complete=report['complete'],matrix_cells=len(cells),new_cells_supplied=len(provided),inherited=a,new=report['new_aggregate'],cost=report['cost']),ensure_ascii=False,indent=2))
    return report


def self_test():
    """Metric/null/subset boundaries, no source data mutation or simulator."""
    base=dict(total_physics_steps=2100,complete=True,label_eligible=True,completed_phases=['close','lift'],task_success=False)
    assert valid_label(base)==0
    assert valid_label(dict(base,total_physics_steps=0)) is None
    assert valid_label(dict(base,label_eligible=False)) is None
    assert valid_label(dict(base,infrastructure_error='failed')) is None
    assert valid_label(dict(base,complete=False)) is None
    assert valid_label(dict(base,task_success=None)) is None
    assert valid_label(dict(base,completed_phases=['close'])) is None
    assert valid_label(dict(base,task_success=True))==1
    assert digest({'b':1,'a':2})==digest({'a':2,'b':1})
    cells=[]
    for o in ['a','b']:
        for g in GESTURES:
            c=dict(object_id=o,gesture_id=g,observed_success_within_max2=(g==1 and o=='a') or (g==2 and o=='b'),actual_attempts=2 if g in [1,2] else 0,terminal=True)
            cells.append(c)
    s=subset_analysis(cells,['a','b'])
    assert s['subset_count']==512 and s['minimum_cardinality_to_cover_observed_union']==2 and s['minimum_cover_subsets']==[[1,2]]
    assert s['leave_one_out_coverage_loss']['1']==['a'] and s['leave_one_out_coverage_loss']['2']==['b']
    assert next(r for r in s['subsets'] if r['gestures']==[2])['additional_objects_over_baseline']==['b']
    assert next(r for r in s['subsets'] if r['gestures']==[])['covered_objects_per_observed_actual'] is None
    print('SELF_TEST_PASS: label/null boundary, canonical hash, 512-subset coverage/cost/unique contribution; no physics')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',nargs='?',type=Path,help='native comparison run root or RESULTS.json');p.add_argument('--run-root',type=Path,help='native comparison root; PREPARED is accepted for a pending snapshot');p.add_argument('--old-pilot',default=str(ROOT/'outputs/boya_pilot8_20261105_v2'));p.add_argument('--cells',type=Path);p.add_argument('--new-progress',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--require-complete',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test();return
    if not args.output_dir:p.error('--output-dir required')
    run(args)

if __name__=='__main__':main()
