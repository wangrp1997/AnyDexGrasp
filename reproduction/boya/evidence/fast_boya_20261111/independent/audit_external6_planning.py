"""Read-only audit: frozen six-object queue, effective planner and commands."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
RUN=ROOT/'outputs/boya_fast6_20261111_v1'
OUT=ROOT/'outputs/boya_fast_audit_20261111_v1'
sys.path.insert(0,str(ROOT/'reproduction/boya'))
import planning_recovery as recovery
from prepare_s40_port_top1_v2 import verify_selected
import execute_collection_queue_v3 as executor

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

protocol=read(RUN/'protocol.json')
queue=read(RUN/'planning/queue_manifest.json')
jobs=read(RUN/'planning/jobs.json')
assert queue['complete'] and not queue['errors'] and queue['inputs_unchanged']
assert queue['external_protocol_ref']==dict(path=str(RUN/'protocol.json'),sha256=sha(RUN/'protocol.json'))
assert protocol['planning_recovery_policy']==recovery.POLICY
assert queue['arm_budget_per_object']==4 and queue['physical_attempts_per_object']==1
assert queue['training_writeback'] is False and queue['benchmark_only'] is True
expected={x['object_id'] for x in protocol['objects']}
assert len(expected)==6 and {x['object_id'] for x in queue['objects']}==expected
integration=ROOT/'reproduction/boya/prepare_boya_fast6_top1.py'
tree=ast.parse(integration.read_text())
assert any(isinstance(n,ast.ImportFrom) and n.module=='planning_recovery' and any(a.name=='prepare_one' for a in n.names) for n in ast.walk(tree))
checked={}
for document in (protocol,queue,jobs):
 for section in ('source_sha256','input_sha256'):
  for p,h in document.get(section,{}).items():
   if p in checked:assert checked[p]==h
   else:assert sha(p)==h;checked[p]=h

def command_audit(row,candidate):
 verify_selected(candidate,row)
 p=Path(row['plan_dir']);plan=read(p/'plan.json');baseline=read(p/'baseline/frozen_input.json')
 frozen=read(p/'frozen_input.json')
 fields=['initial','source_palm','source_object','world_T_object','palm','hand_motor_names','open_motors','closed_motors','dt_s']
 assert all(plan[k]==baseline[k] and plan[k]==frozen[k] for k in fields)
 rec=read(p/'RECOVERY_RESULTS.json')
 assert plan['planning_recovery']['results_sha256']==sha(p/'RECOVERY_RESULTS.json')
 assert plan['planning_recovery']['source_sha256']==sha(recovery.__file__)
 assert plan['planning_recovery']['selected_attempt']==rec['selected']['label']
 assert sha(p/'commands.npz')==row['commands_sha256']==plan['commands_sha256']
 assert sha(p/'path_geometry.json')==plan['path_geometry_sha256']
 assert sha(p/'lift_arm_geometry.json')==plan['lift_arm_geometry_sha256']
 backend,admission,bridge=recovery.backend_modules()
 # Match the production import order; the source model captures hand paths.
 from scripts.research.run_template_arm_plan_sequence import execute_command
 args=backend.parse_args(execute_command(plan['arguments'],OUT/'unlaunched',p)[3:])
 backend.bind_hand(args)
 from canondex.planning.model import PlanningModel
 from canondex.grasp.template_gripper_adapter import TemplateFamily
 from canondex.grasp.template_gripper_closure import TwoSidedClosure
 model=PlanningModel(args.robot_urdf)
 family=TemplateFamily.load(args.package,args.hand_urdf,args.hand_scene,plan['gesture_id'])
 opened,closed=np.asarray(plan['open_motors']),np.asarray(plan['closed_motors'])
 closure=TwoSidedClosure.from_template(family,opened,closed)
 with np.load(args.candidates/'candidates.npz',allow_pickle=False) as source:
  source_palm=source['proposal_palm'][0];source_object=source['world_T_object']
  target_object=np.asarray(read(args.world_object_pose))
  target_palm,placement,rotation=bridge.place_grasp_at_object(source_palm,source_object,target_object)
  direction=rotation@source['proposal_approach_direction'][0]
  assert np.array_equal(source['proposal_q'][0],opened) and np.array_equal(source['proposal_q_close'][0],closed)
  assert np.allclose(target_palm,plan['palm'],atol=1e-14,rtol=0)
  assert np.array_equal(placement,np.asarray(plan['world_T_object']))
 palm=np.asarray(plan['palm']);pre=palm.copy();pre[:3,3]-=.1*direction
 with np.load(p/'commands.npz',allow_pickle=False) as z:
  contract=executor.command_contract(plan,z)
  names=list(z['joint_names']);phase=z['phase']
  assert np.array_equal(z['initial_position'],[plan['initial'][name] for name in names])
  hand=[n for n in model.joints if n not in bridge.ARM_JOINTS and n not in ('FAJ1','FAJ3')]
  assert len(hand)==20
  start={name:int(np.flatnonzero(phase==name)[0]) for name in np.unique(phase)}
  hand_error=0.;target_error=0.;fk_error=np.zeros(2)
  for step,values in enumerate(z['joint_target']):
   state=model.full_state(dict(zip(names,map(float,values))));name=phase[step];k=step-start[name]+1
   motors=closure.command(k/250,k/250) if name=='close' else closed if name in ('hold','lift') else opened
   desired=model.full_state({n:plan['initial'][n] for n in model.active_names}|dict(zip(family.hand.names,map(float,motors))))
   hand_error=max(hand_error,max(abs(state[n]-desired[n]) for n in hand))
   assert state['FAJ1']==plan['initial']['FAJ1'] and state['FAJ3']==plan['initial']['FAJ3']
   if name in ('approach','close','hold','lift'):
    target=palm.copy()
    if name=='approach':target=pre.copy();target[:3,3]+=bridge.smooth_fraction(k/500)*.1*direction
    if name=='lift':target[2,3]+=bridge.smooth_fraction(k/500)*.1
    target_error=max(target_error,float(abs(target-z['desired_palm'][step]).max()))
    fk_error=np.maximum(fk_error,bridge.palm_error(model.fk.transforms(state)['palm'],target))
  assert hand_error==0 and target_error<1e-14
  assert fk_error[0]<=1e-4 and .15*fk_error[1]<=1e-4
  indexes=[names.index(n) for n in bridge.ARM_JOINTS]
  assert np.all(z['joint_target'][:,indexes]>=model.lower[indexes]-1e-7)
  assert np.all(z['joint_target'][:,indexes]<=model.upper[indexes]+1e-7)
  derivatives={}
  for name in ('home_to_pregrasp','approach','lift'):
   ids=np.flatnonzero(phase==name);q=np.vstack([z['joint_target'][ids[0]-1,indexes],z['joint_target'][ids][:,indexes]])
   v=np.gradient(q,.004,axis=0,edge_order=2);a=np.gradient(v,.004,axis=0,edge_order=2);j=np.gradient(a,.004,axis=0,edge_order=2)
   derivatives[name]=dict(velocity_ratio=float(np.max(abs(v)/model.velocity[indexes])),acceleration=float(abs(a).max()),jerk=float(abs(j).max()))
   assert derivatives[name]['velocity_ratio']<=1+1e-8 and derivatives[name]['acceleration']<=3+1e-7 and derivatives[name]['jerk']<=30+1e-7
  path=read(p/'path_geometry.json');lift=read(p/'lift_arm_geometry.json')
  assert path['ok'] and lift['ok'] and not path['violations'] and not lift['violations']
  assert path['samples']==sum(contract['expected_phase_counts'][x] for x in ('settle','home_to_pregrasp','approach'))
  assert lift['samples']==500
 inputs={};checked_trial=executor.base.check_trial(row,inputs)
 return dict(complete=True,trial_id=row['trial_id'],global_score_rank=candidate['row']['global_score_rank'],
  selected_attempt=rec['selected']['label'],baseline_admitted=rec['baseline_admitted'],recovered=rec['recovered'],
  original_verify_selected=True,original_v3_check_trial=True,source_hand_object_initial_exact=True,
  hand20_command_max_error_rad=hand_error,action_palm_target_max_error=target_error,
  commanded_palm_FK_error=dict(position_m=float(fk_error[0]),orientation_rad=float(fk_error[1])),
  temporal_contract=contract,derivatives=derivatives,source_collision_receipts_bound_and_passed=True,
  note='No new IK/path planning, physical scene or full collision query rerun in this audit; FK and command arithmetic only')

catalog={o['object_id']:o for o in jobs['objects']}
objects=[];trials=[];counts=dict(candidate_preparations=0,baseline_pregrasp_queries=0,expanded_pregrasp_queries=0,full_path_attempts=0,named_transit_attempts=0)
for obj in queue['objects']:
 oid=obj['object_id'];items=catalog[oid]['candidates'];attempts=obj['attempts']
 assert len(attempts)==obj['arm_preparations']<=min(4,len(items))
 assert [a['trial_id'] for a in attempts]==[c['job']['trial_id'] for c in items[:len(attempts)]]
 assert all(a['status']!='ready' for a in attempts[:-1])
 assert bool(obj['selected'])==bool(attempts and attempts[-1]['status']=='ready')
 if obj['selected']:assert obj['selected']==attempts[-1]['result']
 else:assert len(attempts)==min(4,len(items))
 sub=[]
 for a,c in zip(attempts,items):
  row=a['result'];p=RUN/'planning/plans'/row['trial_id']/'plan_000';rec=read(p/'RECOVERY_RESULTS.json')
  assert rec['complete'] and rec['policy']==recovery.POLICY and rec['physics_steps']==0
  assert rec['attempts'][0]['label']=='baseline'
  assert bool(rec['selected'])==(row['status']=='ready')
  baseline=read(p/'ORIGINAL_IK_SEARCH.json') if (p/'ORIGINAL_IK_SEARCH.json').exists() else None
  expanded=read(p/'EXPANDED_IK_SEARCH.json') if (p/'EXPANDED_IK_SEARCH.json').exists() else None
  oldq=baseline['queries'] if baseline else read(p/'plan.json')['pregrasp_ik']['motion_seed_selection']['queries']
  newq=expanded['queries'] if expanded else 0
  assert oldq==8 and newq in (0,24)
  if rec['baseline_admitted']:assert newq==0 and len(rec['attempts'])==1
  if expanded:
   assert len(expanded['attempts'])==24 and expanded['per_seed_max_nfev']==300
  assert len({x['label'].split('_')[1] for x in rec['attempts'] if x['label'].startswith('branch_')})<=4
  for x in rec['attempts']:
   if x['route']:
    direct='baseline' if x['label'].startswith('baseline') else x['label'].split('_via_')[0]+'_direct'
    assert recovery.only_home_collisions(p/direct)
  sub.append(dict(trial_id=row['trial_id'],rank=a['global_score_rank'],baseline_admitted=rec['baseline_admitted'],
   recovered=rec['recovered'],selected_attempt=None if not rec['selected'] else rec['selected']['label'],
   baseline_pregrasp_queries=oldq,expanded_pregrasp_queries=newq,path_attempts=len(rec['attempts']),wall_s=rec['wall_s']))
  counts['candidate_preparations']+=1;counts['baseline_pregrasp_queries']+=oldq;counts['expanded_pregrasp_queries']+=newq
  counts['full_path_attempts']+=len(rec['attempts']);counts['named_transit_attempts']+=sum(bool(x['route']) for x in rec['attempts'])
  if row['status']=='ready':trials.append(command_audit(row,c))
 objects.append(dict(object_id=oid,status=obj['status'],geometry_admitted_total=obj['geometry_admitted_total'],attempts=sub,
  baseline_ready_in_executed_prefix=any(x['baseline_admitted'] for x in sub),recovery_ready=any(x['recovered'] for x in sub)))
assert {x['trial_id'] for x in trials}=={x['trial_id'] for x in queue['trials']}
counts['total_pregrasp_queries']=counts['baseline_pregrasp_queries']+counts['expanded_pregrasp_queries']
counts['ready_objects']=len(trials);counts['recovery_ready_objects']=sum(x['recovered'] for x in trials)
result=dict(complete=True,protocol_sha256=sha(RUN/'protocol.json'),queue_sha256=sha(RUN/'planning/queue_manifest.json'),
 integration_calls_declared_prepare_one=True,first_ready_and_four_candidate_budget_passed=True,
 source_and_input_files_verified=len(checked),counts=counts,objects=objects,ready_command_audits=trials,
 pregrasp_query_count_scope='Exact local pregrasp solve calls; approach/lift waypoint solves in failed paths are not completely logged and are not included',
 comparison_scope='Baseline only known for candidates actually visited by recovery-first selection; unvisited later candidates are not evaluated',
 planning_wall_s=queue['wall_s'],additional_planning_calls=0,new_physics_steps=0,
 audit_script_sha256=sha(__file__))
(OUT/'EXTERNAL6_PLANNING_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(complete=True,counts=counts,objects=[dict(object_id=o['object_id'],status=o['status'],recovery_ready=o['recovery_ready']) for o in objects]),indent=2))
