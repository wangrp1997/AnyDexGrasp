"""Frozen three-object Boya hand-only diagnostic, distinct from EGAD12 results.

A3/E0 retain their actually executed candidate for a matched backend contrast.
A6 is the first numbered arm-null object and retains its highest geometry-valid
candidate. No new representation, ranking, fitting or outcome-dependent retry.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, hashlib, html, json, os, subprocess, sys, time
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp'); B=ROOT/'reproduction/boya'
BASE=ROOT/'outputs/egad_external12_20261107_v1'
CASES=(('egad_A6',17,'First numbered previous arm-null object; first geometry-admitted score rank'),
       ('egad_A3',58,'Matched previously executed candidate, not fresh score top1'),
       ('egad_E0',4,'Matched previously executed candidate, not fresh score top1'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p); t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def source_cases():
 q=read(BASE/'planning/queue_manifest.json');d=read(BASE/'diagnostic/RESULTS.json');s=read(BASE/'objects.json')
 assert d['status']=='complete' and q['complete'] and not q['errors']
 docs=[]
 for oid,rank,why in CASES:
  pool=BASE/'diagnostic'/oid/f'pool_{rank:05d}';g=read(pool/'generation.json');record=g['records'][0]
  assert record['global_score_rank']==rank and record['geometry']['ok']
  old=next(x for x in q['objects'] if x['object_id']==oid)
  if oid!='egad_A6':
   assert old['selected']['trial_id']==f'{oid}_port_rank_{rank:05d}'
   plan_dir=Path(old['selected']['plans'][0]['plan_dir'])
  else:
   assert old['selected'] is None
   assert read(BASE/'diagnostic'/oid/'RESULTS.json')['first_admitted']['global_score_rank']==rank
   assert old['attempts'][0]['trial_id']==f'{oid}_port_rank_{rank:05d}'
   plan_dir=None
  row=next(x for x in s['objects'] if x['object_id']==oid)
  z=np.load(pool/'candidates.npz',allow_pickle=False)
  assert np.allclose(z['world_T_object'],row['world_T_object'],atol=1e-14,rtol=0)
  docs.append(dict(object_id=oid,candidate_dir=str(pool),selection_reason=why,
    candidate_rank=rank,candidate_record=record,object_row=row,old_selected_plan=str(plan_dir)if plan_dir else None,
    old_arm_status=old['status'],old_execution=str(BASE/'execution/execution'/oid)if plan_dir else None))
 return docs

def prepare(out):
 if out.exists():raise ValueError('Fresh run directory required')
 docs=source_cases();backend=read(B/'backend.example.json');inputs={};sources={}
 def freeze(p,expected=None,section=inputs):
  p=Path(p).resolve();h=sha(p)
  if expected and h!=expected:raise ValueError('Changed input: '+str(p))
  section[str(p)]=h
 for name in ['protocol.json','objects.json','diagnostic/RESULTS.json','planning/queue_manifest.json','execution/RESULTS.json']:
  freeze(BASE/name)
 training=read(BASE/'protocol.json')
 for p,h in training['input_sha256'].items():freeze(p,h)
 for path in [Path(__file__).resolve(),B/'run_boya_floating_diagnostic.py']:
  freeze(path,section=sources)
 for name in ['template_gripper_closure.py','template_gripper_drive.py','physics.py','tabletop_pickup.py','baselines/isaac_assets.py']:
  freeze(Path('/home/wangrenpeng/CanonDex/canondex/grasp')/name,section=sources)
 for name in ['ur16e_boya.py','hand_actuation.py','kinematics.py']:
  freeze(Path('/home/wangrenpeng/CanonDex/canondex/robots')/name,section=sources)
 for name in ['robot_collision.py','object_contacts.py','grasp_object.py','floor.py','execute_motion.py','runtime.py','gesture_collision.py']:
  freeze(Path('/home/wangrenpeng/CanonDex/canondex/scene')/name,section=sources)
 for key in ['hand_urdf','hand_scene','package','physics_config','diagnostic_config']:freeze(backend[key])
 # Bind every original source hand mesh, including visuals.
 import xml.etree.ElementTree as ET
 hand=Path(backend['hand_urdf'])
 for mesh in ET.parse(hand).findall('.//mesh'):
  freeze(hand.parent/mesh.get('filename'))
 for case in docs:
  p=Path(case['candidate_dir']);g=read(p/'generation.json')
  for name in ['generation.json','candidates.npz','world_object_pose.json']:freeze(p/name)
  for path,h in g['input_sha256'].items():freeze(path,h)
  row=case['object_row']
  for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256')]:freeze(row[pk],row[hk])
  for part in row['convex_parts']:freeze(part['path'],part['sha256'])
  if case['old_selected_plan']:
   for name in ['plan.json','commands.npz']:freeze(Path(case['old_selected_plan'])/name)
 out.mkdir(parents=True);(out/'jobs').mkdir()
 protocol=dict(schema='anydex_boya_hand_only_three_protocol_v1',created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  source_external_run=str(BASE),source_external_protocol_sha256=sha(BASE/'protocol.json'),
  purpose='Separate complete-arm admission/motion from hand/object contact at frozen original candidate poses',
  object_ids=[x['object_id']for x in docs],selection_is_failure_targeted=True,independent_generalization=False,
  cases=[{k:v for k,v in c.items()if k!='object_row'}for c in docs],
  robot='Original Boya palm/finger subtree with three collision-free translational gantry joints; fixed wrist orientation per candidate',
  wrist_drive=dict(stiffness_n_m=100000.,damping_n_s_m=1000.,effort_limit_n=1000.,dummy_link_mass_kg=.001),
  wrist_constraints='No IK, arm links, arm path or reach constraint; no hand/object root state writes after initialization',
  unchanged=['CGR outputs','four scoring heads','score ranks','candidate world hand/object relation','16 hand drive/mimic parameters except existing 0.6Nm thumb-root setting','source visual and convex collision meshes','mass .1kg','friction .6','table plane and collisions','object-only -0.98N gravity','0.004s timestep'],
  changed=['Remove six robot-arm joints and wrist hardware; retain palm/fingers','Three explicit strong-PD translation joints replace arm motion','Initial 800 steps hold open at pregrasp instead of settle50 plus home-to-pregrasp750'],
  phases_steps=dict(prehold=800,approach=500,close=250,hold=50,lift=500),pregrasp_distance_m=.1,lift_distance_m=.1,
  closure_policy='Original TwoSidedClosure.from_template, unchanged original open and nominal endpoint; no tactile adjustment',
  physical_attempts_per_case=1,outcome_dependent_retry=False,training_writeback=False,new_CGR_forwards=0,new_training_updates=0,
  original_success_rate_retained='1/12 remains original full-arm external result; these three are selected diagnostic cases',
  source_sha256=sources,input_sha256=inputs)
 save(out/'protocol.json',protocol);psha=sha(out/'protocol.json')
 for case in docs:
  job=dict(case,**{k:backend[k]for k in ['hand_urdf','hand_scene','package','physics_config','diagnostic_config']},
    output_dir=str(out/'execution'/case['object_id']),protocol_path=str(out/'protocol.json'),protocol_sha256=psha)
  save(out/'jobs'/(case['object_id']+'.json'),job)
 save(out/'PROGRESS.json',dict(status='prepared',object_ids=protocol['object_ids'],protocol_sha256=psha,actual_grasps=0))
 return protocol

def execute(out,workers=3):
 p=read(out/'protocol.json')
 for section in ['source_sha256','input_sha256']:
  for f,h in p[section].items():
   if sha(f)!=h:raise ValueError('Frozen dependency changed: '+f)
 if (out/'RESULTS.json').exists():raise ValueError('Execution already completed; no repeat')
 if (out/'DISPATCH.json').exists():raise ValueError('Dispatch already recorded; do not restart blindly')
 save(out/'DISPATCH.json',dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),objects=p['object_ids'],workers=workers))
 start=time.monotonic();results={};errors=[]
 def one(index,oid):
  log=out/(oid+'.physics.log');argv=['taskset','-c',f'{32+8*index}-{39+8*index}',sys.executable,'-B',str(B/'run_boya_floating_diagnostic.py'),'--job',str(out/'jobs'/(oid+'.json'))]
  before=time.monotonic()
  env=os.environ.copy();env['CANONDEX_COLLISION_PARTITION_CACHE']=str(ROOT/'outputs/boya_collision_partition_cache')
  with log.open('xb')as f:rc=subprocess.run(argv,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600).returncode
  target=out/'execution'/oid/'evaluation.json'
  if rc or not target.exists():raise RuntimeError(f'{oid} process rc={rc}; see {log}')
  value=read(target)
  if not value['complete']:raise RuntimeError('Incomplete '+oid)
  return dict(object_id=oid,evaluation=str(target),evaluation_sha256=sha(target),task_success=value['task_success'],physical_steps=value['total_physics_steps'],wall_s=time.monotonic()-before,process_returncode=rc)
 with ThreadPoolExecutor(max_workers=workers)as pool:
  tasks={pool.submit(one,i,oid):oid for i,oid in enumerate(p['object_ids'])}
  for f in as_completed(tasks):
   oid=tasks[f]
   try:results[oid]=f.result()
   except Exception as exc:errors.append(dict(object_id=oid,error=repr(exc)))
   save(out/'PROGRESS.json',dict(status='executing',finished=len(results),results=list(results.values()),errors=errors))
   print('FLOAT_CASE',oid,results.get(oid),flush=True)
 report=dict(schema='anydex_boya_hand_only_three_results_v1',status='complete'if not errors else'infrastructure_error',protocol_sha256=sha(out/'protocol.json'),
  results=[results[i]for i in p['object_ids']if i in results],errors=errors,actual_grasps=len(results),
  success=sum(v['task_success']for v in results.values()),physical_steps=sum(v['physical_steps']for v in results.values()),
  benchmark_only=True,training_writeback=False,independent_generalization=False,wall_s=time.monotonic()-start)
 for section in ['source_sha256','input_sha256']:
  for f,h in p[section].items():
   if sha(f)!=h:raise ValueError('Dependency changed during run: '+f)
 save(out/'RESULTS.json',report);save(out/'PROGRESS.json',report);return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['prepare','execute'],required=True);a=p.parse_args()
 v=prepare(a.output.resolve())if a.stage=='prepare'else execute(a.output.resolve());print(json.dumps(dict(stage=a.stage,status=v.get('status','prepared'))),flush=True)
if __name__=='__main__':main()
