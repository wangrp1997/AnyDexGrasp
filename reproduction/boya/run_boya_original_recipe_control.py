"""Execute only first geometry-valid code100 candidates on a controlled wrist.

Four known objects, one physical try each, no training or outcome-based retries.
Scoring comes from frozen original final-ReLU selected-MSE heads; this remains
an explicitly approximate Boya simulator adaptation, not author hardware.
"""
from pathlib import Path
import argparse,sys
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya';sys.path.insert(0,str(B))
import run_egad_floating_three as runner
read=runner.read;save=runner.save;sha=runner.sha
IDS=('egad_A0','egad_A3','egad_A6','egad_E0')
def prepare(base):
 out=base/'physical_control'
 if out.exists():raise ValueError('Fresh physical control required')
 d=read(base/'diagnostic/RESULTS.json')
 if d['status']!='complete' or d['head_recipe']!='code100_original_selected_mse' or d['approved_object_ids']!=list(IDS):raise ValueError('Complete frozen code100 known4 ranking required')
 old=read(ROOT/'outputs/egad_floating_three_20261108_v1/protocol.json');sources=dict(old['source_sha256']);sources[str(Path(__file__).resolve())]=sha(__file__)
 inputs=dict(d['input_sha256']);sources.update(d['source_sha256'])
 rows={r['object_id']:r for r in read(base/'objects.json')['objects']};backend=read(B/'backend.example.json');cases=[]
 for oid in IDS:
  receipt=base/'diagnostic'/oid/'RESULTS.json';a=read(receipt)['first_admitted'];inputs[str(receipt)]=sha(receipt)
  if a is None:raise ValueError('No candidate; do not silently replace object')
  pool=Path(a['candidate_dir']);g=read(pool/'generation.json')
  for f in [pool/'generation.json',pool/'candidates.npz',pool/'world_object_pose.json']:
   inputs[str(f)]=sha(f)
  for p,h in g['input_sha256'].items():inputs[p]=h
  row=rows[oid]
  for p,h in [(row['mesh_path'],row['mesh_sha256']),(row['urdf_path'],row['urdf_sha256']),(row['pose_path'],row['pose_sha256'])]+[(x['path'],x['sha256'])for x in row['convex_parts']]:inputs[p]=h
  cases.append(dict(object_id=oid,candidate_rank=a['global_score_rank'],candidate_dir=str(pool),object_row=row,
   old_execution=None,selection_reason='First complete visible-cloud geometry-admitted candidate in the frozen original-MSE ranking; no arm/reach filtering'))
 for sec in [sources,inputs]:
  for p,h in sec.items():
   if sha(p)!=h:raise ValueError('Source/input changed '+p)
 out.mkdir();(out/'jobs').mkdir()
 protocol=dict(schema='anydex_boya_original_recipe_controlled_wrist_v1',object_ids=list(IDS),cases=cases,
  source_sha256=sources,input_sha256=inputs,head_recipe=d['head_recipe'],evaluation_role='known_object_recipe_alignment',
  complete_author_hardware_reproduction=False,independent_generalization=False,physical_attempts=4,outcome_dependent_retry=False,
  training_writeback=False,closure='Original250-step nominal Boya closure, same motor/mimic properties as all frozen gantry baselines',
  allowed_change='Frozen code100 original-MSE scoring heads select a new geometry-valid candidate; original RGB-D/CGR, calibration, domain, source objects, coarse proxy geometry and native physics gates retained',
  arm_present=False,models_updated=False,new_CGR_forwards=0)
 save(out/'protocol.json',protocol)
 for case in cases:
  job=dict(case,**{k:backend[k]for k in ['hand_urdf','hand_scene','package','physics_config','diagnostic_config']},output_dir=str(out/'execution'/case['object_id']),protocol_path=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'))
  save(out/'jobs'/(case['object_id']+'.json'),job)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=prepare(a.run.resolve());runner.execute(out,workers=4)
