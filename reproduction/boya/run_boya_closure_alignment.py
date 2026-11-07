"""Four paired published-position-timing controls, no model/grasp changes.

A0 is the positive control; A3/A6/E0 retain previous failed coarse-geometry
candidates. This imports the published Allegro TEN POSITION TARGET schedule on
Boya, not Allegro kinematics/hardware or tactile/torque-feedback control.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,json,os,subprocess,sys,time
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya';sys.path.insert(0,str(B))
import run_egad_floating_three as prior
read=prior.read;save=prior.save;sha=prior.sha
CASES=('egad_A0','egad_A3','egad_A6','egad_E0')
def run(out):
 if out.exists():raise ValueError('Fresh directory required')
 source=ROOT/'outputs/egad_floating_three_20261108_v1';positive=ROOT/'outputs/boya_contact_proxy_alignment_20261108_v2'
 originals={oid:source/'jobs'/(oid+'.json')for oid in CASES if oid!='egad_A0'}
 originals['egad_A0']=positive/'jobs/egad_A0_coarse.json'
 p0=read(source/'protocol.json');inputs=dict(p0['input_sha256']);sources=dict(p0['source_sha256']);jobs=[]
 for path in [Path(__file__).resolve(),B/'run_boya_allegro_step_control.py']:
  sources[str(path)]=sha(path)
 official=ROOT/'ur_toolbox/ur_toolbox/robot/Allegro/Allegro.py';sources[str(official)]=sha(official)
 for oid in CASES:
  jp=originals[oid];job=read(jp);inputs[str(jp)]=sha(jp)
  # A0 baseline protocol adds input hashes omitted by the old 3-case set.
  if oid=='egad_A0':
   for path,h in read(positive/'protocol.json')['input_sha256'].items():inputs[path]=h
  job.update(object_id=oid,source_object_id=oid,output_dir=str(out/'execution'/oid),
    closure_control='published_allegro_ten_position_steps_v1')
  jobs.append(job)
 for path,h in {**inputs,**sources}.items():
  if sha(path)!=h:raise ValueError('Frozen source changed '+path)
 out.mkdir();(out/'jobs').mkdir()
 protocol=dict(schema='anydex_boya_position_timing_alignment_v1',object_ids=list(CASES),source_sha256=sources,input_sha256=inputs,
  cases=[dict(object_id=j['object_id'],original_job=str(originals[j['object_id']]),old_execution=None)for j in jobs],
  selection='All3 old controlled-wrist failures plus A0 successful coarse control; all same coarse assets and exact original candidates',
  allowed_change='Only250 close-phase finger position targets:10 equal position increments held100ms each instead of250 held4ms each',
  unchanged=['All desired palm targets','All other phase commands','Boya hand endpoints','Original coarse collision geometry','Object initial pose/mass/friction','Motor/mimic properties','1s close plus.2s hold','CGR and scoring weights'],
  mismatch_remaining='Author Allegro starts from actual measured opening; this timing-only control starts from frozen opening endpoint',
  physical_attempts=4,outcome_dependent_retries=False,independent_generalization=False,training_writeback=False)
 save(out/'protocol.json',protocol)
 for job in jobs:
  job.update(protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'));save(out/'jobs'/(job['object_id']+'.json'),job)
 save(out/'DISPATCH.json',dict(objects=list(CASES),actual_grasps=0));start=time.monotonic();results={};errors=[]
 def one(i,oid):
  cmd=['taskset','-c',f'{32+8*i}-{39+8*i}',sys.executable,'-B',str(B/'run_boya_allegro_step_control.py'),'--job',str(out/'jobs'/(oid+'.json'))]
  env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
  t=time.monotonic()
  with(out/(oid+'.physics.log')).open('xb')as log:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
  target=out/'execution'/oid/'evaluation.json'
  if rc or not target.exists():raise RuntimeError(f'{oid}rc={rc}')
  e=read(target)
  if not e['complete']:raise RuntimeError('Incomplete '+oid)
  return dict(object_id=oid,evaluation=str(target),evaluation_sha256=sha(target),task_success=e['task_success'],physical_steps=e['total_physics_steps'],wall_s=time.monotonic()-t)
 with ThreadPoolExecutor(max_workers=4)as pool:
  tasks={pool.submit(one,i,oid):oid for i,oid in enumerate(CASES)}
  for f in as_completed(tasks):
   oid=tasks[f]
   try:results[oid]=f.result()
   except Exception as exc:errors.append(dict(object_id=oid,error=repr(exc)))
   save(out/'PROGRESS.json',dict(status='executing',results=list(results.values()),errors=errors));print('CLOSE_ALIGN',oid,results.get(oid,errors),flush=True)
 for path,h in {**inputs,**sources}.items():
  if sha(path)!=h:raise ValueError('Frozen dependency changed during execution '+path)
 report=dict(status='complete'if not errors else'infrastructure_error',protocol_sha256=sha(out/'protocol.json'),results=[results[x]for x in CASES if x in results],errors=errors,
  physical_steps=sum(x['physical_steps']for x in results.values()),actual_grasps=len(results),success=sum(x['task_success']for x in results.values()),
  wall_s=time.monotonic()-start,benchmark_only=True,training_writeback=False,independent_generalization=False)
 save(out/'RESULTS.json',report);save(out/'PROGRESS.json',report);return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
