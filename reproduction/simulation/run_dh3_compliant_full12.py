"""Complete the original twelve scenes with one fixed compliant simulation proxy.

Reuse the two already executed banana scenes only after exact model, inputs,
assumptions and script binding. Keep original no-candidate cases in denominator.
Execute each remaining original top1 once, without new inference or tuning.
"""
from __future__ import annotations
import argparse,concurrent.futures as cf,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON='/home/wangrenpeng/boya-inhand-wm/.venv/bin/python'
OLD=ROOT/'outputs/dh3_real_mesh_generalization_20261103_v1/run/RESULTS.json'
REUSED=ROOT/'outputs/dh3_compliant_comparison_20261104_v3'
RUNNER=ROOT/'reproduction/simulation/run_compliant_dh3_refined_scene.py'
EXPECTED_MODEL='50e02b4d8a7696c5507a27acb92a20b06c81195f75fde8322380ec7999580aec'
EXPECTED_RUNNER='cdc86125eb944446442d396a369aef44333ca2b87d902a72bd150ccc44f96349'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(r):assert sha(r['path'])==r['sha256'],r['path']

def prepare(out,workers):
 assert not out.exists(),'Retain all previous campaigns'
 assert sha(ROOT/'reproduction/simulation/dh3_compliant_proxy.py')==EXPECTED_MODEL
 assert sha(RUNNER)==EXPECTED_RUNNER
 old=read(OLD);assert len(old['scenes'])==12
 sources=[rec(p) for p in [Path(__file__),RUNNER,ROOT/'reproduction/simulation/dh3_compliant_proxy.py',ROOT/'reproduction/simulation/dh3_time_refinement_smoke.py']]
 cases=[]
 for row in old['scenes']:
  scene=rec(row['scene_manifest']);assert scene['sha256']==row['scene_manifest_sha256']
  inference=rec(Path(row['inference_output_dir'])/'RESULTS.json');assert inference['sha256']==row['inference_RESULTS_sha256']
  case=dict(scene_id=row['scene_id'],scene=scene,inference=inference,original_status=row['status'],original_collision_stages=row['collision_stages'])
  if not row['final_top1_selected']:
   assert row['collision_stages']['collision_admitted']==0
   case.update(mode='no_original_candidate',physics_output_dir=None,reason=row['not_entered_physics_reason'])
  else:
   selection=rec(row['selection_results']);assert selection['sha256']==row['selection_sha256']
   candidate=rec(row['selected_candidate_npz']);assert candidate['sha256']==row['selected_candidate_sha256']
   baseline=Path(row['physics_output_dir'])
   case.update(mode='execute',selection=selection,candidate=candidate,baseline_result=rec(baseline/'RESULTS.json'),baseline_trace=rec(baseline/'trace.npz'),physics_output_dir=str(out/'physics'/row['scene_id']))
   if row['scene_id'] in ('banana_yaw000','banana_yaw090'):
    path=REUSED/'physics'/row['scene_id'];controller=read(path/'CONTROLLER.json');result=read(path/'RESULTS.json')
    assert controller['controller_script_sha256']==EXPECTED_MODEL and result['runner_sha256']==EXPECTED_RUNNER
    assert controller['selection_sha256']==selection['sha256'] and controller['candidate_sha256']==candidate['sha256']
    assert controller['scene_manifest_sha256']==scene['sha256'] and result['actual_time_step_s']==1/3840 and result['physics_steps']==39360
    assert controller['compliance']['branch_stiffness_Nm_rad']==.8 and controller['compliance']['branch_damping_Nm_s_rad']==.02 and controller['compliance']['shaft_inertia_kg_m2']==.001
    case.update(mode='reuse',physics_output_dir=str(path),reused_result=rec(path/'RESULTS.json'),reused_controller=rec(path/'CONTROLLER.json'),reused_trace=rec(path/'trace.npz'))
  cases.append(case)
 assert sum(c['mode']=='execute' for c in cases)==7
 assert sum(c['mode']=='reuse' for c in cases)==2
 assert sum(c['mode']=='no_original_candidate' for c in cases)==3
 plan=dict(schema='DH3-fixed-compliant-full12-v1',source_cohort=rec(OLD),reuse_source_cohort=rec(REUSED/'RESULTS.json'),source_bindings=sources,cases=cases,workers=workers,
  model_parameters=dict(k_Nm_rad=.8,d_Nm_s_rad=.02,J_kg_m2=.001,actual_hz=3840,common_shaft_motor_cap_Nm=.6,spread_motor_cap_Nm=.3),
  functional_outcome='Original lift_observed: final COM lift >50mm, final lowest native AABB >40mm, hand-object contact in final half-second; report separately from old numeric gates',
  numeric_diagnostics='Retain original velocity<1.5rad/s, gear residual<.01rad, wrist error<2mm and finite force budgets without rewriting any old results',
  attempts_per_selected_scene=1,network_forwards=0,candidate_reselection=False,parameter_search=False,all_twelve_in_denominator=True,
  scope='Known twelve-scene diagnostic set with original algorithm candidates and declared approximate execution, not an unseen test or paper hardware benchmark')
 out.mkdir(parents=True);(out/'physics').mkdir();(out/'logs').mkdir();shutil.copyfile(__file__,out/'launcher_source_frozen.py');save(out/'PLAN.json',plan)
 return plan

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 assert 1<=a.workers<=7
 out=a.output_dir.resolve();plan=prepare(out,a.workers);started=time.monotonic();rows={}
 for case in plan['cases']:
  if case['mode']=='no_original_candidate':
   rows[case['scene_id']]=dict(scene_id=case['scene_id'],mode=case['mode'],functional_status='no_original_candidate',lift_observed=False,port_stability_pass=None,physics_steps=0,new_physics_steps=0,physics_output_dir=None,object_final_lift_m=None)
  elif case['mode']=='reuse':
   result=read(Path(case['physics_output_dir'])/'RESULTS.json')
   rows[case['scene_id']]=dict(result,scene_id=case['scene_id'],mode='reuse',functional_status='lifted_and_held' if result['lift_observed'] else 'not_held',new_physics_steps=0,physics_output_dir=case['physics_output_dir'])
 def progress(final=False):
  ordered=[rows[c['scene_id']] for c in plan['cases'] if c['scene_id'] in rows]
  p=dict(status='completed' if final else 'running',completed_scenes=len(ordered),total_scenes=12,lifted_and_held=sum(bool(r['lift_observed']) for r in ordered),numeric_gate_passes=sum(bool(r['port_stability_pass']) for r in ordered),scenes=ordered,wall_seconds=time.monotonic()-started)
  temp=out/'PROGRESS.tmp';save(temp,p);temp.replace(out/'PROGRESS.json')
  return p
 progress()
 def run(case):
  for r in plan['source_bindings']:verify(r)
  for key in ('scene','inference','selection','candidate','baseline_result','baseline_trace'):verify(case[key])
  log=out/'logs'/(case['scene_id']+'.log');t=time.monotonic()
  with log.open('w') as f:
   completed=subprocess.run([PYTHON,'-B',str(RUNNER),'--scene-manifest',case['scene']['path'],'--selection-results',case['selection']['path'],'--output-dir',case['physics_output_dir']],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  path=Path(case['physics_output_dir'])/'RESULTS.json'
  result=read(path) if path.exists() else dict(status='execution_error',lift_observed=False,port_stability_pass=False,physics_steps=0)
  return dict(result,scene_id=case['scene_id'],mode='execute',functional_status=('execution_error' if completed.returncode else ('lifted_and_held' if result['lift_observed'] else 'not_held')),returncode=completed.returncode,new_physics_steps=result['physics_steps'],wall_seconds_process=time.monotonic()-t,physics_output_dir=case['physics_output_dir'],log=str(log))
 jobs=[c for c in plan['cases'] if c['mode']=='execute']
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  pending={pool.submit(run,c):c for c in jobs}
  for future in cf.as_completed(pending):
   c=pending[future]
   try:r=future.result()
   except Exception as e:r=dict(scene_id=c['scene_id'],mode='execute',functional_status='dispatch_error',lift_observed=False,port_stability_pass=False,physics_steps=0,new_physics_steps=0,error=repr(e),physics_output_dir=c['physics_output_dir'])
   rows[c['scene_id']]=r;progress();print(json.dumps({k:r.get(k) for k in ['scene_id','functional_status','object_final_lift_m','port_stability_pass','physics_steps','wall_seconds_process','error']}),flush=True)
 result=progress(True)
 result.update(scope=plan['scope'],plan=rec(out/'PLAN.json'),new_physics_steps=sum(r['new_physics_steps'] for r in rows.values()),reused_physics_steps=sum(r['physics_steps'] for r in rows.values() if r['mode']=='reuse'),total_represented_physics_steps=sum(r['physics_steps'] for r in rows.values()),new_physical_seconds=sum(r['new_physics_steps'] for r in rows.values())/3840,represented_physical_seconds=sum(r['physics_steps'] for r in rows.values())/3840,new_grasp_attempts=7,reused_grasp_attempts=2,no_original_candidate_scenes=3,attempts_per_selected_scene=1,network_forwards=0,candidate_reselection=False,old_numeric_diagnostics_unchanged=True,functional_success_denominator=12,functional_success_numerator=sum(bool(r['lift_observed']) for r in rows.values()),functional_success_rate=sum(bool(r['lift_observed']) for r in rows.values())/12,source_original_8_of_12_unchanged=True)
 save(out/'RESULTS.json',result);print(json.dumps({k:result[k] for k in ['status','functional_success_numerator','functional_success_denominator','numeric_gate_passes','new_physics_steps','new_physical_seconds','wall_seconds']}),flush=True)
if __name__=='__main__':main()
