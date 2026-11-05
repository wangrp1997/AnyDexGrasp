#!/usr/bin/env python3
"""Two-worker zero-step native asset preparation for the frozen 40+10 selection.

Runs the unchanged source-preserving native cooker. No observation, grasp,
network or training. External result paths can be adopted by pipeline v2.
"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

EXPECTED_SELECTION_SHA256='487017083e79eb1323f243946ddf2b931dac6bd87d86f52632a254ba5226f60d'
NATIVE=Path('/home/wangrenpeng/CanonDex/scripts/research/preprocess_native_object.py')
CANON=NATIVE.parents[2]
IMPL=('canondex/grasp/native_object_preprocess.py','canondex/grasp/collision_refinement.py',
 'canondex/grasp/convex_partition.py','canondex/grasp/physics.py','canondex/scene/convex_cooking.py',
 'scripts/research/preprocess_native_object.py','scripts/research/template_gripper_native_audit.py',
 'scripts/research/run_template_arm_positive.py')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify(mapping):
 for path,digest in mapping.items():
  if sha(path)!=digest:raise ValueError('Frozen asset/source changed: '+str(path))


def freeze(args,repo):
 if sha(args.selection)!=EXPECTED_SELECTION_SHA256:raise ValueError('Require the existing fixed selection; no new selection')
 selection=read(args.selection);objects=selection['objects'];old=read(args.pilot_selection)['objects']
 if len(objects)!=50 or Counter(o['split']for o in objects)!={'train':40,'validation':10}:raise ValueError('Expected frozen 40train/10validation')
 if len({o['mesh_sha256']for o in objects})!=50:raise ValueError('Repeated mesh SHA group')
 ids={o['object_id']for o in objects};oldids={o['object_id']for o in old}
 oldmeshes={sha(Path(o['object_root'])/'mesh/simplified.obj')for o in old}
 if len(old)!=8 or ids&oldids or {o['mesh_sha256']for o in objects}&oldmeshes:raise ValueError('Pilot8 overlap in identity or mesh group')
 train={o['mesh_sha256']for o in objects if o['split']=='train'};val={o['mesh_sha256']for o in objects if o['split']=='validation'}
 if train&val:raise ValueError('Train/validation mesh groups overlap')
 backend=read(args.backend_config);physics=Path(backend['physics_config']).resolve();files={}
 sys.path.insert(0,str(CANON))
 from canondex.grasp.physics import load_physics_config
 profile=load_physics_config(physics)
 profile_sha=jsha(profile.to_dict())
 def bind(path,digest=None):
  path=str(Path(path).resolve());h=sha(path)
  if digest is not None and h!=digest:raise ValueError('Declared frozen asset differs: '+path)
  files[path]=h
 for p in [args.selection,args.pilot_selection,args.backend_config,physics,Path(__file__)]:bind(p)
 if args.resume_from:
  for p in [args.resume_from/'PROTOCOL.json',args.resume_from/'SUMMARY.json']:bind(p)
  old_protocol=read(args.resume_from/'PROTOCOL.json')
  if old_protocol['selection_sha256']!=EXPECTED_SELECTION_SHA256:raise ValueError('Resume asset selection differs')
 for rel in IMPL:bind(CANON/rel)
 for obj in objects:
  if obj.get('preflight_status')!='ready' or obj['scale']!=.1 or obj['object_root'].split('/')[-1]!=obj['object_id']:raise ValueError('Invalid fixed source asset row')
  for pk,hk in [('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')]:bind(obj[pk],obj[hk])
  for part in obj['convex_parts']:bind(part['path'],part['sha256'])
 return dict(schema='anydex_boya_collection_native_preflight_v2',selection=str(args.selection),selection_sha256=EXPECTED_SELECTION_SHA256,
  pilot_selection=str(args.pilot_selection),backend_config=str(args.backend_config),physics_config=str(physics),physics_config_sha256=sha(physics),physics_profile_sha256=profile_sha,physics_profile_definition=profile.to_dict(),
  resume_from=str(args.resume_from)if args.resume_from else None,
  objects=[{k:o[k]for k in ['object_id','split','object_root','scale','mesh_sha256','mesh_group_sha256']}for o in objects],
  output_dir=str(args.output_dir),cache_dir=str(args.output_dir/'native_object_cache'),workers=args.workers,
  max_failed_partitions=32,tolerance_m=20e-6,source_union_preserved=True,
  identity_checks=dict(fixed50mesh_groups=True,train40_validation10=True,train_validation_mesh_disjoint=True,pilot8_id_and_mesh_disjoint=True),
  input_sha256=files,new_physics_steps=0,new_network_forwards=0,new_training_steps=0,
  scope='Only existing fifty assets, exact source convex union and unchanged 20um native readback. Independent dynamic execution-scene audit still required.')


def validate(obj,resultfile,protocol):
 d=read(resultfile)
 if not(d.get('complete') and d.get('original_or_refined_within_20um') and d.get('trial_physics_steps')==0):raise ValueError('Incomplete/nonzero-step native asset receipt')
 binding=d['binding'];b=binding['inputs']
 if binding['key']!=jsha(b):raise ValueError('Native cache identity differs')
 if b['object_id']!=obj['object_id'] or b['scale']!=obj['scale'] or b['physics_profile_sha256']!=protocol['physics_profile_sha256'] or b['max_failed_partitions']!=32 or b['tolerance_m']!=20e-6:raise ValueError('Native asset binding does not match this frozen collection')
 verify({str(CANON/p):h for p,h in b['implementations'].items()})
 verify({str(Path(obj['object_root'])/p):h for p,h in b['source_files'].items()})
 cache=Path(d['cache_directory']);artifacts=[str(resultfile)]
 for rel,h in d['artifact_sha256'].items():
  verify({str(cache/rel):h});artifacts.append(str(cache/rel))
 repair=d.get('repair_path')
 if bool(repair)!=d['repair_required']:raise ValueError('Repair receipt/path inconsistent')
 if repair:
  repairdoc=read(repair)
  if repairdoc['object_id']!=obj['object_id'] or repairdoc['physics_profile_sha256']!=protocol['physics_profile_sha256'] or repairdoc['tolerance_m']!=20e-6:raise ValueError('Repair recipe bound to wrong object/profile/tolerance')
  if str(Path(repair))not in artifacts:artifacts.append(str(Path(repair)))
 return dict(object_id=obj['object_id'],split=obj['split'],status='repaired'if repair else'ready',
  result_json=str(resultfile),result_sha256=sha(resultfile),repair_path=repair,repair_sha256=sha(repair)if repair else None,
  artifacts=artifacts,cache_directory=str(cache),cache_key=binding['key'],source_union_preserved=True,
  source_partitions=d['source_partitions'],refined_partitions=d['refined_partitions'],failed_partitions=len(d['failed_partitions']),
  source_volume_sum_m3=d['source_volume_sum_m3'],refined_volume_sum_m3=d['refined_volume_sum_m3'],
  trial_physics_steps=0,native_tolerance_m=20e-6,max_failed_partitions=32)


def main():
 repo=Path(__file__).resolve().parents[2]
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--selection',type=Path,default=repo/'outputs/collection40_20261105_v1/selection_v2/object_selection.json')
 p.add_argument('--pilot-selection',type=Path,default=repo/'outputs/boya_pilot8_20261105_v1/object_selection.json')
 p.add_argument('--backend-config',type=Path,default=repo/'outputs/boya_pilot8_20261105_v1/backend_pilot.json')
 p.add_argument('--output-dir',type=Path,default=repo/'outputs/collection40_20261106_v2/assets_preflight_v2')
 p.add_argument('--resume-from',type=Path,help='Read-only completed native results from an earlier wrapper segment')
 p.add_argument('--workers',type=int,choices=[1,2],default=2);a=p.parse_args()
 for k in ['selection','pilot_selection','backend_config','output_dir']:setattr(a,k,getattr(a,k).resolve())
 if a.resume_from:a.resume_from=a.resume_from.resolve()
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/'preflight.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  protocol_path=a.output_dir/'PROTOCOL.json';proposed=freeze(a,repo)
  if protocol_path.exists():
   protocol=read(protocol_path)
   if protocol!=proposed:raise ValueError('Frozen native preflight differs; use new output directory')
  else:protocol=proposed;save(protocol_path,protocol)
  verify(protocol['input_sha256']);started=time.monotonic();rows=protocol['objects'];jobs={o['object_id']:dict(object_id=o['object_id'],split=o['split'],status='queued')for o in rows}
  report=dict(schema=protocol['schema'],status='running',pid=os.getpid(),protocol=str(protocol_path),protocol_sha256=sha(protocol_path),
   selection_sha256=protocol['selection_sha256'],started_at_unix=time.time(),max_concurrent_workers=a.workers,
   max_failed_partitions=32,tolerance_m=20e-6,source_union_preserved=True,total_objects=50,results=[],jobs=jobs,errors=[],
   new_physics_steps=0,new_network_forwards=0,new_training_steps=0)
  def publish():
   report.update(elapsed_wall_s=time.monotonic()-started,ready_objects=sum(j['status']=='ready'for j in jobs.values()),
    repaired_objects=sum(j['status']=='repaired'for j in jobs.values()),completed_objects=sum(j['status']in['ready','repaired']for j in jobs.values()))
   report['results']=[jobs[o['object_id']]for o in rows if jobs[o['object_id']]['status']in['ready','repaired']]
   save(a.output_dir/'PROGRESS.json',report)
  publish()
  save(a.output_dir/'ADOPTION_CONTRACT.json',dict(native_preflight=str(a.output_dir),protocol=str(protocol_path),protocol_sha256=sha(protocol_path),
   summary=str(a.output_dir/'SUMMARY.json'),result_pattern=str(a.output_dir/'OBJECT/result.json'),cache_dir=protocol['cache_dir'],
   results_schema='SUMMARY.results: object_id,result_json,result_sha256,repair_path,repair_sha256,artifacts',
   selection_sha256=protocol['selection_sha256'],max_failed_partitions=32,tolerance_m=20e-6,trial_physics_steps=0,
   adoption='Validate frozen selection/profile/scale/source/implementation/artifact SHA. Write run-local SUMMARY referencing these original result/cache paths; no need to launch fifty new native scenes.'))
  def one(obj):
   output=a.output_dir/obj['object_id'];resultfile=output/'result.json';reused=resultfile.exists();reused_from=None
   if not reused and a.resume_from and(a.resume_from/obj['object_id']/'result.json').is_file():
    previous=a.resume_from/obj['object_id']/'result.json'
    validate(obj,previous,protocol)
    output.mkdir(exist_ok=False);resultfile.write_bytes(previous.read_bytes());reused=True;reused_from=str(previous)
   if not reused:
    if output.exists():raise ValueError('Partial native object output retained: '+str(output))
    command=[sys.executable,'-B',str(NATIVE),'--object-root',obj['object_root'],'--scale',str(obj['scale']),
     '--physics-config',protocol['physics_config'],'--cache-dir',protocol['cache_dir'],'--output-dir',str(output),'--max-failed-partitions','32']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMNI_KIT_ACCEPT_EULA='YES',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',CANONDEX_COLLISION_PARTITION_CACHE=str(a.output_dir/'collision_partition_cache'))
    with output.with_suffix('.log').open('xb')as log:completed=subprocess.run(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    if completed.returncode:raise RuntimeError('Native source-preserving preprocess failed for '+obj['object_id']+' exit '+str(completed.returncode))
   result=validate(obj,resultfile,protocol);result['reused_result']=reused;result['reused_from']=reused_from;return result
  pending=iter(rows);active={};stopped=False
  with ThreadPoolExecutor(max_workers=a.workers)as exe:
   def dispatch():
    try:o=next(pending)
    except StopIteration:return False
    jobs[o['object_id']]['status']='running';active[exe.submit(one,o)]=o;publish();return True
   for _ in range(a.workers):dispatch()
   while active:
    done,_=wait(active,return_when=FIRST_COMPLETED)
    for f in done:
     obj=active.pop(f);oid=obj['object_id']
     try:
      jobs[oid]=f.result();print(json.dumps(dict(object_id=oid,status=jobs[oid]['status'],result_json=jobs[oid]['result_json'])),flush=True)
     except Exception as error:
      stopped=True;jobs[oid].update(status='infrastructure_error',error=repr(error));report['errors'].append(dict(object_id=oid,error=repr(error)));print(json.dumps(report['errors'][-1]),flush=True)
     publish()
    if not stopped:
     while len(active)<a.workers and dispatch():pass
  if stopped:
   for j in jobs.values():
    if j['status']=='queued':j['status']='not_dispatched_after_infrastructure_error'
  try:verify(protocol['input_sha256']);report['frozen_sources_unchanged']=True
  except Exception as error:stopped=True;report['frozen_sources_unchanged']=False;report['errors'].append(dict(stage='final_input_SHA',error=repr(error)))
  report.update(status='infrastructure_stopped'if stopped else'complete',complete=not stopped,finished_at_unix=time.time());publish()
  save(a.output_dir/'SUMMARY.json',{k:v for k,v in report.items()if k!='jobs'})
  print(json.dumps({k:v for k,v in report.items()if k not in['jobs','results']},indent=2),flush=True)
  if stopped:raise SystemExit(1)


if __name__=='__main__':main()
