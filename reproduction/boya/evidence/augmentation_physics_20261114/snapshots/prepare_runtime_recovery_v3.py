"""Keep the failed zero-step dispatch; recover the SAME frozen queue with an audit-only dependency overlay."""
from pathlib import Path
import sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
from run_augmented_cohort import read,save,sha
r=Path(__file__).resolve().parent;old=r/'physical_history_v2';out=r/'physical_runtime_v3'
assert not out.exists()
result=read(old/'RESULTS.json');assert result['status']=='infrastructure_error' and result['new_attempts']==result['new_physics_steps']==0
for p in (old/'execution').glob('*'):
 assert read(p/'initialization_audit.json')['physics_steps']==0
 assert not (p/'evaluation.json').exists() and not (p/'commands.npz').exists() and not (p/'execution_progress.json').exists()
assert read(r/'zero_step_diagnostic_v3/SUCCESS.json')['physics_steps']==0
p=read(old/'protocol.json');out.mkdir()
for name in ('ALL_CHOICES_FREEZE.json','STUDY_PROTOCOL.json'):(out/name).write_bytes((old/name).read_bytes())
provenance=ROOT/'outputs/native_deps/clarabel_0111_wheel/PROVENANCE.json';dep=read(provenance);assert dep['official_sha256_verified']
p.update(runtime_recovery=dict(reason='Missing clarabel 0.11.1 in native zero-step cooking audit; original CLI masks exception during app.close',
 previous_protocol=str(old/'protocol.json'),previous_protocol_sha256=sha(old/'protocol.json'),previous_dispatch_new_physics_steps=0,
 selection_commands_physics_and_executor_unchanged=True,pythonpath_overlay=str(ROOT/'outputs/native_deps/clarabel_0111_overlay'),
 no_completed_or_partial_physical_attempt_repeated=True),concurrency=4)
for f in [Path(__file__),old/'RESULTS.json',old/'DISPATCH.json',old/'protocol.json',r/'zero_step_diagnostic_v2/FAILURE.json',r/'zero_step_diagnostic_v3/SUCCESS.json',provenance,Path(dep['wheel']),*[f for f in (old/'execution').glob('*/initialization_audit.json')],*[f for f in (old/'logs').glob('*.log')]]:
 p['input_sha256'][str(f)]=sha(f)
p['input_sha256'].update(dep['files']);p['source_sha256'][str(Path(__file__))]=sha(__file__)
for f in (Path('/home/wangrenpeng/CanonDex/canondex/grasp/cooking_audit.py'),Path('/home/wangrenpeng/CanonDex/scripts/research/template_gripper_native_audit.py')):
 p['source_sha256'][str(f)]=sha(f);p['input_sha256'][str(f)]=sha(f)
for c in p['cases']:
 if c['kind']=='new':c['job_path']=str(out/'jobs'/(c['object_id']+'.json'))
save(out/'protocol.json',p);jobs={}
for c in p['cases']:
 if c['kind']=='new':
  job=read(old/'jobs'/(c['object_id']+'.json'));job.update(output_dir=str(out/'execution'/c['object_id']),protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'))
  path=Path(c['job_path']);save(path,job);jobs[str(path)]=sha(path)
save(out/'PREPARED.json',dict(jobs_sha256=jobs,protocol_sha256=sha(out/'protocol.json'),new_attempts=7,unique=14,reused=7,no_geometry=0))
print('RUNTIME_RECOVERY_PREPARED: SAME 7 new / 7 reused',flush=True)
