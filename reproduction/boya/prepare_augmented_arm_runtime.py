"""Native import and unchanged settling initialization for six frozen new scenes.

No hand, score, training or grasp execution. Each external object settles in
its own CPU scene so another object's instability cannot decide its outcome.
"""
from pathlib import Path
import argparse,copy,json,subprocess,sys,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');B=ROOT/'reproduction/boya';C=Path('/home/wangrenpeng/CanonDex');sys.path.insert(0,str(B))
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from run_collection_pipeline_v2 import sha,read,save,verify,native_profile_identity
from settle_s40_scenes import POLICY
NATIVE=C/'scripts/research/preprocess_native_object.py'
OBJECT_IDS=('egad_A2','egad_A5','egad_C2','egad_C5','egad_E2','egad_E5')
PROTOCOL_SCHEMA='anydex_boya_augmented_arm6_protocol_v1'


def validate_fast6_protocol(protocol):
 if (protocol.get('schema')!=PROTOCOL_SCHEMA or protocol.get('object_count')!=6
     or [r.get('object_id')for r in protocol.get('objects',[])]!=list(OBJECT_IDS)):
  raise ValueError('Exact frozen six-object protocol and identity order required')
 if (protocol.get('physical_attempts_per_object')!=1 or protocol.get('arm_budget_per_object')!=4
     or protocol.get('outcome_dependent_retry')is not False or protocol.get('benchmark_only')is not True
     or protocol.get('training_writeback')is not False or protocol.get('head_recipe')!='paired_frozen_original_and_preactivation'):
  raise ValueError('Frozen benchmark-only first-ready/arm4/one-attempt policy required')
 if (protocol.get('source_unit_to_m')!=.001 or protocol.get('object_mass_kg')!=.1
     or protocol.get('object_friction')!=.6 or protocol.get('settling_policy')!=POLICY):
  raise ValueError('Original uniform scale/mass/friction/settling policy required')
 if not protocol.get('source_sha256')or not protocol.get('input_sha256'):
  raise ValueError('Explicit frozen implementation and input maps required')
 return list(OBJECT_IDS)


def validate_fast6_manifest(scene,protocol):
 ids=validate_fast6_protocol(protocol)
 if (scene.get('schema')!='anydex_boya_external_scene_manifest_v1'
     or [r.get('object_id')for r in scene.get('objects',[])]!=ids
     or scene.get('approved_object_ids')!=ids or scene.get('active_object_count')!=6):
  raise ValueError('Explicit frozen six-object external scene manifest required')
 for row in scene['objects']:
  if row.get('id')!=row['object_id'] or row.get('split')!='external_validation' or row.get('scale')!=1:
   raise ValueError('External scene identity/split/metre export scale changed')
  if row.get('preflight_status')!='ready':raise ValueError('Asset preparation incomplete; preserve original IDs')
  if any(row.get(k)is not None for k in ['label','training_label','success','result','trial_outcome']):
   raise ValueError('New external scenes must not carry grasp outcomes')
 return scene['objects'],ids

def command(argv,log):
 t=time.monotonic()
 env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
 with Path(log).open('xb')as f:p=subprocess.run(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
 return dict(argv=list(map(str,argv)),returncode=p.returncode,wall_s=time.monotonic()-t,log=str(log))

def run(a):
 root=a.run_dir.resolve();scene=read(a.scene_manifest);protocol=read(root/'protocol.json');pref=dict(path=str(root/'protocol.json'),sha256=sha(root/'protocol.json'))
 rows,ids=validate_fast6_manifest(scene,protocol)
 verify(protocol['source_sha256']);verify(protocol['input_sha256'])
 backend_path=ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json';backend=read(backend_path);physics=backend['physics_config'];profile=native_profile_identity(physics)
 inputs=dict(protocol['input_sha256']);inputs[str(a.scene_manifest.resolve())]=sha(a.scene_manifest);inputs[str(root/'protocol.json')]=sha(root/'protocol.json')
 for r in rows:
  for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256')]:inputs[r[pk]]=r[hk]
  for part in r['convex_parts']:inputs[part['path']]=part['sha256']
 # The outer protocol explicitly pins the unchanged backend dependency set.
 # No directory glob: concurrently authored opt-in entries are unrelated.
 sources=dict(protocol['source_sha256'])
 for p in [Path(__file__).resolve(),B/'settle_s40_scenes.py',B/'run_collection_pipeline_v2.py',NATIVE]:sources[str(p)]=sha(p)
 verify(inputs);verify(sources)
 nd=root/'native_preprocessing';sp=nd/'SUMMARY.json'
 if not sp.exists():
  nd.mkdir(exist_ok=False);save(nd/'PLAN.json',dict(external_protocol_ref=pref,source_sha256=sources,input_sha256=inputs,objects=ids,new_physics_steps=0,workers=a.workers))
  def native(r):
   target=nd/r['object_id'];receipt=command([sys.executable,'-B',NATIVE,'--object-root',r['object_root'],'--scale','1','--physics-config',physics,'--cache-dir',root/'native_object_cache','--output-dir',target,'--max-failed-partitions','32'],nd/(r['object_id']+'.log'))
   if receipt['returncode']:raise RuntimeError('Native preprocessing failed '+r['object_id']+'; see '+receipt['log'])
   rp=target/'result.json';d=read(rp);bound=d['binding']['inputs']
   if not(d.get('complete')and d.get('trial_physics_steps')==0 and d.get('original_or_refined_within_20um')):raise ValueError('Native20um gate failed')
   if bound['object_id']!=r['object_id']or bound['scale']!=1 or bound['physics_profile_sha256']!=profile or bound['max_failed_partitions']!=32 or bound['tolerance_m']!=20e-6:raise ValueError('Foreign native binding')
   verify({str(Path(r['object_root'])/k):h for k,h in bound['source_files'].items()});verify({str(C/k):h for k,h in bound['implementations'].items()})
   cache=Path(d['cache_directory']);refs=[rp]
   for name,h in d['artifact_sha256'].items():verify({str(cache/name):h});refs.append(cache/name)
   repair=d.get('repair_path')
   if repair:refs.append(Path(repair))
   return dict(object_id=r['object_id'],result_json=str(rp),repair_path=repair,repair_sha256=sha(repair)if repair else None,artifacts=list(map(str,refs)),process=receipt)
  result={};errors=[]
  with ThreadPoolExecutor(max_workers=a.workers)as pool:
   pending={pool.submit(native,r):r for r in rows}
   for f in as_completed(pending):
    oid=pending[f]['object_id']
    try:result[oid]=f.result()
    except Exception as exc:errors.append(dict(object_id=oid,error=repr(exc)))
    save(nd/'PROGRESS.json',dict(finished=len(result),errors=errors,objects=6));print('EXTERNAL_NATIVE '+oid,flush=True)
  save(sp,dict(complete=not errors,results=[result[i]for i in ids if i in result],errors=errors,new_physics_steps=0,physics_profile_sha256=profile,external_protocol_ref=pref))
 native=read(sp)
 if not native['complete']:raise RuntimeError('Native external assets incomplete; no test outcomes fabricated')
 if a.phase=='native':return native
 sd=root/'settling';sd.mkdir(exist_ok=False);nativeby={r['object_id']:r for r in native['results']}
 for r in native['results']:
  for p in r['artifacts']:inputs[p]=sha(p)
 specs=[]
 for r in rows:
  p=sd/(r['object_id']+'.WORK.json');value=dict(objects=[r],output=str(sd/r['object_id']),native=nativeby,physics_config=physics,inputs=inputs,sources=sources,policy=POLICY);save(p,value);specs.append(p)
 save(sd/'PLAN.json',dict(schema='anydex_boya_augmented_arm6_initialization_v1',external_protocol_ref=pref,policy=POLICY,source_sha256=sources,input_sha256=inputs,workers=a.workers,objects=ids,grasp_attempts=0))
 def settle(p):
  rec=command([sys.executable,'-B',B/'settle_s40_scenes.py','--worker-spec',p],p.with_suffix('.log'));out=Path(read(p)['output'])
  if rec['returncode']:raise RuntimeError('Settling failed: '+rec['log'])
  return read(out/'RESULTS.json'),rec
 got={};errors=[];ticks=0;steps=0;processes=[]
 with ThreadPoolExecutor(max_workers=a.workers)as pool:
  jobs={pool.submit(settle,p):p for p in specs}
  for f in as_completed(jobs):
   p=jobs[f]
   try:
    value,rec=f.result();processes.append(rec);ticks+=value['physics_ticks'];steps+=value['object_steps'];got.update({r['object_id']:r for r in value['objects']})
   except Exception as exc:errors.append(dict(spec=str(p),error=repr(exc)))
   save(sd/'PROGRESS.json',dict(finished=len(got),errors=errors,objects=6));print('EXTERNAL_SETTLE '+p.stem,flush=True)
 final=dict(schema='anydex_boya_augmented_arm6_initialization_results_v1',complete=not errors and len(got)==6,objects=[got[i]for i in ids if i in got],ready_objects=sum(r['ready']for r in got.values()),errors=errors,physics_ticks=ticks,object_steps=steps,grasp_attempts=0,source_sha256=sources,input_sha256=inputs,external_protocol_ref=pref,processes=processes)
 save(sd/'RESULTS.json',final);verify(inputs);verify(sources)
 if not final['complete']or final['ready_objects']!=6:raise RuntimeError('Initialization incomplete; frozen objects retained; no fallback placement')
 manifest=copy.deepcopy(scene)
 for row in manifest['objects']:
  s=got[row['object_id']];T=np.asarray(s['world_T_object']);q=np.asarray(s['measured_settled_pose_xyzw'][3:]);pose=sd/'poses'/(row['object_id']+'.json');pose.parent.mkdir(exist_ok=True);save(pose,[[*T[:3,3],q[3],*q[:3]]])
  row.update(source_pre_settle_pose_path=row['pose_path'],pose_path=str(pose),pose_sha256=sha(pose),world_T_object=T.tolist(),world_pose_xyzw=s['measured_settled_pose_xyzw'],settling_ready=True,settling=s)
  mesh=trimesh.load_mesh(row['mesh_path'],process=False);mesh.apply_transform(T);row.update(world_AABB_m=mesh.bounds.tolist(),world_extents_m=np.ptp(mesh.vertices,axis=0).tolist())
 manifest.update(external_protocol_ref=pref,source_scene_manifest=str(a.scene_manifest.resolve()),source_scene_manifest_sha256=sha(a.scene_manifest),settling_provenance=dict(complete=True,ready_objects=6,results_json=str(sd/'RESULTS.json'),results_sha256=sha(sd/'RESULTS.json'),entry_boundary='Measured settled pose, independent cold recreation with zero velocity; not continuous scene',policy=POLICY),initialization_object_steps=steps,initialization_physics_ticks=ticks,grasp_attempts=0)
 save(root/'objects.json',manifest);print(json.dumps(dict(status='complete',objects=6,initialization_object_steps=steps,grasp_attempts=0)),flush=True)
 return final
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=4,choices=range(1,5));p.add_argument('--phase',choices=['all','native'],default='all');run(p.parse_args())
