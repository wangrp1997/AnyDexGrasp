"""Pre-execution correction: include the omitted exact A0 MSE history.

All sixteen frozen selections and fourteen unique poses remain unchanged.
The first preparation remains intact with zero dispatch and zero physics.
"""
from pathlib import Path
import sys,copy,json
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
from run_augmented_cohort import read,save,sha,same_pose,eligible,baseline
r=Path(__file__).resolve().parent;out=r/'physical_history_v2'
assert not (r/'DISPATCH.json').exists() and not out.exists()
p=read(r/'protocol.json');old=ROOT/'outputs/boya_original_recipe_alignment_20261108_v1/physical_control/jobs/egad_A0.json';j=read(old)
cid='egad_A0_identity_original_mse';case=next(c for c in p['cases']if c['object_id']==cid)
assert case['kind']=='new' and same_pose(case['candidate_dir'],j['candidate_dir']) and eligible(j,read(baseline('egad_A0')))
out.mkdir();(out/'ALL_CHOICES_FREEZE.json').write_bytes((r/'ALL_CHOICES_FREEZE.json').read_bytes())
(out/'STUDY_PROTOCOL.json').write_bytes((r/'STUDY_PROTOCOL.json').read_bytes())
case.update(kind='reused',job_path=str(old),matching_history=[str(old)])
p['new_queue'].remove(cid);p.update(new_attempts=7,reused=7,maximum_new_physics_steps=14700,
 preparation_correction='Include omitted A0 MSE exact history. No selection, ranking, configuration or execution changes.',
 original_preparation=dict(path=str(r/'protocol.json'),sha256=sha(r/'protocol.json'),dispatched=False,new_physics_steps=0))
for c in p['conditions']:
 if c['condition_id']==cid:c['kind']='reused'
for f in [Path(__file__),r/'protocol.json',r/'PREPARED.json',old,Path(j['protocol_path']),*[Path(j['output_dir'])/n for n in ('evaluation.json','commands.npz','trace.npz','initialization_audit.json','native_audit.json')]]:
 p['input_sha256'][str(f)]=sha(f)
p['source_sha256'][str(Path(__file__))]=sha(__file__)
for c in p['cases']:
 if c['kind']=='new':c['job_path']=str(out/'jobs'/(c['object_id']+'.json'))
save(out/'protocol.json',p);jobs={}
for c in p['cases']:
 if c['kind']=='new':
  job=read(r/'jobs'/(c['object_id']+'.json'));job.update(output_dir=str(out/'execution'/c['object_id']),protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'))
  path=Path(c['job_path']);save(path,job);jobs[str(path)]=sha(path)
save(out/'PREPARED.json',dict(jobs_sha256=jobs,protocol_sha256=sha(out/'protocol.json'),new_attempts=7,unique=14,reused=7,no_geometry=0))
save(r/'PREEXECUTION_HISTORY_CORRECTION.json',dict(status='complete',original_preparation=str(r/'protocol.json'),corrected_preparation=str(out/'protocol.json'),correction='Omitted A0 exact history added to history matching; outcome not used to select a pose',sixteen_choices_unchanged=True,fourteen_physical_poses_unchanged=True,new_attempts_before=8,new_attempts_after=7,original_preparation_physics_steps=0,new_CGR_forwards=0,new_head_inference=0,new_geometry_checks=0))
print('HISTORY_CORRECTION_COMPLETE 7 new / 7 reused',flush=True)
