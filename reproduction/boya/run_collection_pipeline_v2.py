"""Version 2: selected calibrated gestures on the unchanged 40/10 mesh split.

--dry-run freezes the input/command plan only: no Kit, model import or physics.
Explicit --phase execute/all enables the original source-bound physical queue.
"""
from __future__ import annotations
import argparse,collections,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path

ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
NATIVE=Path('/home/wangrenpeng/CanonDex/scripts/research/preprocess_native_object.py')
DEFAULTS=dict(selection=str(ROOT/'outputs/collection40_20261105_v1/selection_v2/object_selection.json'),calibration=str(ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json'),backend_config=str(ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json'),checkpoint='/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18',native_deps=str(ROOT/'outputs/native_deps'),representation_device='cuda:0',physics_device='cpu',execution_mode='serial',batch_size=16,execution_workers=2,native_preflight=None,gesture_ids=[1,6,7,8],geometry_workers=2,plan_workers=4,candidate_limit=500,keep=8,max_arm_plans=4,seed=20261105)
STAGES=('native_assets','observations','representation','geometry','planning','execution')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def immutable(path,data):
 if Path(path).exists():
  if read(path)!=data:raise ValueError('Frozen manifest differs: '+str(path))
 else:save(path,data)
def hashes(paths):return{str(Path(p).resolve()):sha(p)for p in dict.fromkeys(map(str,paths))}
def verify(files):
 for path,digest in files.items():
  if not Path(path).is_file()or sha(path)!=digest:raise ValueError('Frozen source/artifact changed: '+path)
def env():return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))


def validate_gestures(values):
 if not isinstance(values,list)or not 1<=len(values)<=4 or any(type(g)is not int or not 1<=g<=10 for g in values)or len(values)!=len(set(values)):
  raise ValueError('Require one to four distinct integer gesture IDs in1..10; four depths give at most16plans/object')

def validate_config(cfg):
 validate_gestures(cfg['gesture_ids'])
 if cfg['physics_device']!='cpu'or cfg['execution_mode']!='serial':raise ValueError('v2 requires CPU serial resident object queues')
 for key,lo,hi in(('execution_workers',1,2),('batch_size',1,16),('geometry_workers',1,2),('plan_workers',1,8)):
  if type(cfg[key])is not int or not lo<=cfg[key]<=hi:raise ValueError('Invalid '+key)

def slot_contract(rows,gestures):
 validate_gestures(gestures);counts=collections.Counter(r['split']for r in rows)
 return dict(slot_count=len(rows)*len(gestures)*4,split_slot_counts={s:n*len(gestures)*4 for s,n in counts.items()},
  valid_label_target_totals={s:len(gestures)*n for s,n in [('train',100),('validation',20)]},per_object_slot_count=len(gestures)*4)

def bounded_jobs(items,workers,fn,on_result=None,on_dispatch=None):
 """Stop new submission after the first infrastructure failure; drain inflight."""
 iterator=iter(items);active={};rows=[];failed=False
 with ThreadPoolExecutor(max_workers=workers)as executor:
  def fill():
   while not failed and len(active)<workers:
    try:item=next(iterator)
    except StopIteration:return
    active[executor.submit(fn,item)]=item
    if on_dispatch:on_dispatch(item,list(active.values()))
  fill()
  while active:
   finished,_=wait(active,return_when=FIRST_COMPLETED)
   for future in finished:
    item=active.pop(future)
    try:value=future.result();bad=isinstance(value,dict)and bool(value.get('infrastructure_error'))
    except Exception:value=dict(infrastructure_error=True,error=traceback.format_exc());bad=True
    failed=failed or bad;row=dict(item=item,value=value,failed=bad);rows.append(row)
    if on_result:on_result(item,value,bad)
   fill()
 return rows

def execution_usable(doc,expected_ids,returncode):
 rows=doc.get('trials',[]);ids=[r.get('trial_id')for r in rows]
 return bool(not returncode and not doc.get('infrastructure_error')and not doc.get('error')
  and doc.get('status')in('complete','complete_with_null')and doc.get('input_sha256_unchanged')is True
  and doc.get('source_sha256_unchanged')is True and len(ids)==len(set(ids))and set(ids)==set(expected_ids)
  and all(r.get('label')in(0,1)and r.get('label_eligible')is True and r.get('physics_steps')==2100 for r in rows))

def native_profile_identity(path):
 # Native receipts bind the normalized profile, separately from file-byte SHA.
 canon=str(NATIVE.parents[2])
 if canon not in sys.path:sys.path.insert(0,canon)
 from canondex.grasp.physics import load_physics_config
 from canondex.grasp.collision_refinement import profile_identity
 return profile_identity(load_physics_config(path))

def external_native_refs(directory,objects,selection_sha256,profile_sha256):
 path=Path(directory)/'SUMMARY.json'
 if not path.is_file():raise ValueError('Declared external native preflight is not complete; no duplicate preprocessing')
 doc=read(path);rows=doc.get('results',[]);expected={o['object_id']for o in objects};actual=[r['object_id']for r in rows]
 if not doc.get('complete')or doc.get('errors')or len(actual)!=len(set(actual))or set(actual)!=expected:raise ValueError('External native preflight incomplete or object coverage differs')
 if doc.get('selection_sha256')!=selection_sha256 or doc.get('new_physics_steps')!=0:raise ValueError('External preflight selection or physical cost differs')
 verify({doc['protocol']:doc['protocol_sha256']});protocol=read(doc['protocol'])
 if protocol.get('physics_profile_sha256')!=profile_sha256 or protocol.get('selection_sha256')!=selection_sha256:raise ValueError('External preflight normalized profile or selection differs')
 for row in rows:
  verify({row['result_json']:row['result_sha256']})
  if row.get('repair_path'):verify({row['repair_path']:row['repair_sha256']})
 return {r['object_id']:r for r in rows},[path,Path(doc['protocol'])]

def split_records(records):
 grouped={'train':[],'validation':[]};seen=set();object_splits={}
 for record in records:
  context=record['collection_context'];tid=context['trial_id'];oid=context['object_id'];split=context['split']
  if split not in grouped or tid in seen:raise ValueError('Duplicate record or invalid frozen split')
  if oid in object_splits and object_splits[oid]!=split:raise ValueError('Object crosses train/validation split')
  seen.add(tid);object_splits[oid]=split;grouped[split].append(record)
 return grouped

def execution_costs(rows,ready_ids):
 byid={r['trial_id']:r for r in rows}
 if len(byid)!=len(rows):raise ValueError('Duplicate executor trial identities')
 not_dispatched=[tid for tid in ready_ids if tid not in byid or byid[tid].get('null_reason')=='queue_not_dispatched_after_infrastructure_error']
 unknown=[tid for tid,r in byid.items()if tid not in not_dispatched and r.get('label')is None and not r.get('physics_steps',0)]
 return dict(actual_grasps=sum(r.get('physics_steps',0)>0 for r in rows),physical_steps=sum(r.get('physics_steps',0)for r in rows),not_dispatched_ready_trial_ids=not_dispatched,unknown_execution_trial_ids=unknown)

def label_counts(protocol,rows):
 targets=protocol['valid_label_targets_per_gesture'];byid={j['trial_id']:j for j in protocol['slots']}
 counts={s:{str(g):dict(labels=0,positive=0,negative=0,target=targets[s])for g in protocol['gesture_ids']}for s in('train','validation')}
 if len({r['trial_id']for r in rows})!=len(rows):raise ValueError('Repeated physical trial identity')
 for row in rows:
  job=byid[row['trial_id']]
  if row.get('label')in(0,1)and row.get('label_eligible')is True:
   c=counts[job['split']][str(job['gesture_id'])];c['labels']+=1;c['positive'if row['label']==1 else'negative']+=1
 for split in counts:
  for c in counts[split].values():c.update(shortfall=max(0,c['target']-c['labels']),extra_valid_labels_retained=max(0,c['labels']-c['target']))
 return counts

def freeze(run,args):
 old=read(run/'protocol.json')if(run/'protocol.json').exists()else None
 if old and old.get('schema')!='anydex_boya_collection_pipeline_v2':raise ValueError('v2 requires its own versioned protocol/output directory')
 cfg={k:getattr(args,k)if getattr(args,k)is not None else(old['config'][k]if old else v)for k,v in DEFAULTS.items()}
 for key in('selection','calibration','backend_config','checkpoint','native_deps'):cfg[key]=str(Path(cfg[key]).resolve())
 if cfg['native_preflight']:cfg['native_preflight']=str(Path(cfg['native_preflight']).resolve())
 validate_config(cfg)
 if old:
  if old['config']!=cfg:raise ValueError('Frozen options changed; create a new run')
  verify(old['input_sha256']);verify(old['source_sha256']);verify(old['manifest_sha256']);return old
 if run.exists()and any(run.iterdir()):raise ValueError('New output must be empty; existing experiments are never overwritten')
 run.mkdir(parents=True,exist_ok=True)
 selection=read(cfg['selection']);rows=selection['objects'];counts=collections.Counter(r['split']for r in rows)
 if counts!={'train':40,'validation':10}or len(rows)!=50:raise ValueError('Require frozen40train/10validation objects')
 if len({r['mesh_sha256']for r in rows})!=50 or any(r.get('preflight_status')!='ready'for r in rows):raise ValueError('Unique source mesh groups and ready asset preflight required')
 for key in('selection','calibration','backend_config','checkpoint','native_deps'):cfg[key]=str(Path(cfg[key]).resolve())
 files={}
 def add(p,h=None):
  p=Path(p).resolve();actual=sha(p)
  if h is not None and h!=actual:raise ValueError('Declared source SHA differs: '+str(p))
  files[str(p)]=actual
 for key in('selection','calibration','backend_config','checkpoint'):add(cfg[key])
 for policy in(BOYA/'COLLECTION_PROTOCOL_20261106.md',BOYA/'evidence/collection_20261106/selection_static_audit.json'):add(policy)
 if cfg['native_preflight']:
  for name in('ADOPTION_CONTRACT.json','PROTOCOL.json'):
   path=Path(cfg['native_preflight'])/name
   if path.exists():add(path)
 backend=read(cfg['backend_config']);cal=read(cfg['calibration'])
 for v in backend.values():
  if isinstance(v,str)and v.startswith('/')and Path(v).is_file():add(v)
 for p in Path(backend['reference_root']).rglob('*.json'):add(p)
 for p,meta in cal['source_lock']['source_files'].items():add(p,meta['sha256'])
 for p,h in cal['source_lock']['readonly_fk_dependencies'].items():add(p,h)
 for g in cfg['gesture_ids']:
  if not any(s['gesture_id']==g and s['self_collision_clear']and s['opposing_group']=='ff'for s in cal['slots']):raise ValueError('Missing clear calibration datum for gesture '+str(g))
 # Only source assets/control enter the freeze, never actual grasp outcomes.
 for row in rows:
  for pathkey,hashkey in(('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')):add(row[pathkey],row[hashkey])
  for part in row['convex_parts']:add(part['path'],part['sha256'])
 source_names=['render_collection_observations.py','render_pilot_observations.py','run_collection_representation.py','run_pilot_representation.py','build_collection_pool.py','prepare_collection_queue.py','execute_collection_queue.py','asset_adapter.py','trial_record.py','native_arm.py']
 sources=[Path(__file__).resolve(),BOYA/'run_collection_pipeline.py',*(BOYA/n for n in source_names),NATIVE,ROOT/'reproduction/runtime/run_official_rgbd_frame.py',ROOT/'robot_dh3.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'utils/pt_utils.py']
 canon=NATIVE.parents[2]
 sources.extend(canon/n for n in('canondex/grasp/native_object_preprocess.py','canondex/grasp/collision_refinement.py','canondex/grasp/convex_partition.py','canondex/grasp/physics.py','canondex/scene/convex_cooking.py'))
 rebased=copy.deepcopy(selection)
 for row in rebased['objects']:
  row.update(id=row['object_id'],observation_json=str(run/'observations'/row['object_id']/'observation.json'),representation_dir=str(run/'representations'/row['object_id']))
  row['observation']=row['observation_json'];row['pools']={str(g):{str(d):str(run/'pools'/row['object_id']/('g'+str(g))/('d'+str(d)))for d in range(4)}for g in cfg['gesture_ids']}
 immutable(run/'objects.json',rebased)
 immutable(run/'representation_manifest.json',dict(objects=[{k:r[k]for k in('object_id','observation_json','representation_dir','split')}for r in rebased['objects']]))
 slots=[dict(trial_id=r['object_id']+'__g%02d__d%d'%(g,d),object_id=r['object_id'],split=r['split'],gesture_id=g,extra_depth_index=d,candidate_dir=r['pools'][str(g)][str(d)],observation_json=r['observation_json'],source_npz=str(Path(r['representation_dir'])/'raw_camera_candidates.npz'),calibration=cfg['calibration'],max_arm_plans=cfg['max_arm_plans'])for r in rebased['objects']for g in cfg['gesture_ids']for d in range(4)]
 contract=slot_contract(rows,cfg['gesture_ids'])
 protocol=dict(schema='anydex_boya_collection_pipeline_v2',config=cfg,selection_sha256=files[cfg['selection']],input_sha256=files,source_sha256=hashes(sources),manifest_sha256=hashes([run/'objects.json',run/'representation_manifest.json']),selection_pool_contract_inherited=selection.get('pool_contract'),pool_contract_amendment=dict(reason='Versioned four-gesture expansion after the nine-gesture development comparison. Original selector and v1 protocol remain unchanged; this protocol declares its gesture set and all four depths before collection.',gesture_ids=cfg['gesture_ids'],extra_depth_indices=[0,1,2,3],extra_depth_m=[0.,.01,.02,.03]),gesture_ids=cfg['gesture_ids'],extra_depth_indices=[0,1,2,3],extra_depth_m=[0.,.01,.02,.03],slots=slots,slot_count=contract['slot_count'],split_slot_counts=contract['split_slot_counts'],valid_label_target_totals=contract['valid_label_target_totals'],per_object_slot_limit=contract['per_object_slot_count'],previous_pipeline_source=dict(path=str(BOYA/'run_collection_pipeline.py'),sha256=sha(BOYA/'run_collection_pipeline.py')),valid_label_targets_per_gesture={'train':100,'validation':20},native_asset_policy=dict(max_failed_partitions=32,physics_steps=0,source_union_preserved=True,unchanged_native_tolerance_m=20e-6),scope='Declared versioned collection gestures, not a best-gesture claim; one RGB-D and one shared original identity CGR forward per object. One first arm-admitted actual trial per gesture/depth slot, finite source score order, no outcome feedback or invented labels. Validation is adaptation-development heldout, not unseen-CGR-pretraining.',created_unix_s=time.time())
 save(run/'protocol.json',protocol);return protocol


class Pipeline:
 def __init__(self,run,protocol):
  self.run=run;self.p=protocol;self.c=protocol['config'];self.objects=read(run/'objects.json')['objects'];self.log=run/'logs';self.log.mkdir(exist_ok=True);(run/'stages').mkdir(exist_ok=True)
  self.progress=read(run/'pipeline_progress.json')if(run/'pipeline_progress.json').exists()else dict(schema='anydex_boya_pipeline_progress_v2',stages={},slot_count=protocol['slot_count'],physics_steps=0)
 def check(self):verify(self.p['input_sha256']);verify(self.p['source_sha256']);verify(self.p['manifest_sha256'])
 def publish(self,status):self.progress.update(status=status,updated_unix_s=time.time());save(self.run/'pipeline_progress.json',self.progress)
 def command(self,tag,argv,allowed_failure=False):
  verify(self.p['source_sha256']);start=time.monotonic()
  with(self.log/(tag+'.log')).open('xb')as log:r=subprocess.run(argv,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT)
  verify(self.p['source_sha256'])
  if r.returncode and not allowed_failure:raise RuntimeError('Infrastructure failure '+tag+'; logs/outputs retained')
  return dict(argv=argv,returncode=r.returncode,wall_s=time.monotonic()-start)
 def parallel(self,items,workers,fn):
  rows=bounded_jobs(items,workers,fn)
  if any(r['failed']for r in rows):raise RuntimeError('Infrastructure error; no new dispatch after first failure; finite in-flight tasks drained: '+str([r['value']for r in rows if r['failed']]))
  return [r['value']for r in rows]
 def stage(self,name,fn):
  receipt=self.run/'stages'/(name+'.json');self.check()
  if receipt.exists():
   old=read(receipt)
   if old.get('complete')is not True or old.get('protocol_sha256')!=sha(self.run/'protocol.json'):raise ValueError('Incomplete or foreign stage retained: '+name)
   verify(old['artifacts_sha256']);self.progress['stages'][name]=dict(status='verified_reused',receipt_sha256=sha(receipt));self.publish(name+'_reused');return old['result']
  self.publish(name+'_running');began=time.monotonic();result,artifacts=fn();self.check()
  doc=dict(complete=True,protocol_sha256=sha(self.run/'protocol.json'),artifacts_sha256=hashes(artifacts),wall_s=time.monotonic()-began,result=result)
  save(receipt,doc);self.progress['stages'][name]=dict(status='complete',receipt_sha256=sha(receipt));self.publish(name+'_complete');return result
 def native(self):
  output=self.run/'native_preprocessing';output.mkdir(exist_ok=True)
  physics_path=read(self.c['backend_config'])['physics_config'];profile_sha=native_profile_identity(physics_path)
  external=Path(self.c['native_preflight'])if self.c['native_preflight']else None
  refs,external_assets=external_native_refs(external,self.objects,self.p['selection_sha256'],profile_sha)if external else({},[])
  def one(row):
   target=(external if external else output)/row['object_id']
   argv=[sys.executable,'-B',str(NATIVE),'--object-root',row['object_root'],'--scale',str(row['scale']),'--physics-config',read(self.c['backend_config'])['physics_config'],'--cache-dir',str(self.run/'native_object_cache'),'--output-dir',str(target),'--max-failed-partitions','32']
   resultfile=Path(refs[row['object_id']]['result_json'])if external else target/'result.json'
   if not resultfile.exists():
    if external:raise ValueError('Declared external native preflight is not complete; no duplicate preprocessing: '+str(target))
    if target.exists():raise ValueError('Partial native preprocessing preserved: '+str(target))
    self.command('native_'+row['object_id'],argv)
   d=read(resultfile)
   if not(d.get('complete')and d.get('trial_physics_steps')==0 and d.get('original_or_refined_within_20um')):raise ValueError('Native original20um gate did not pass')
   bound=d['binding']['inputs']
   if bound['object_id']!=row['object_id']or bound['scale']!=row['scale']or bound['physics_profile_sha256']!=profile_sha or bound['max_failed_partitions']!=32 or bound['tolerance_m']!=20e-6:raise ValueError('Native asset binding differs')
   for relative,h in bound['implementations'].items():verify({str(NATIVE.parents[2]/relative):h})
   for relative,h in bound['source_files'].items():verify({str(Path(row['object_root'])/relative):h})
   cache=Path(d['cache_directory']);assets=[resultfile]
   for relative,h in d.get('artifact_sha256',{}).items():verify({str(cache/relative):h});assets.append(cache/relative)
   repair=d.get('repair_path')
   if repair:assets.append(Path(repair))
   return dict(object_id=row['object_id'],result_json=str(resultfile),adopted_external_preflight=external is not None,repair_path=repair,repair_sha256=sha(repair)if repair else None,artifacts=list(map(str,assets)))
  result=self.parallel(self.objects,2,one);summary=output/'SUMMARY.json';save(summary,dict(complete=True,results=result,max_failed_partitions=32,new_physics_steps=0,physics_config_sha256=sha(physics_path),physics_profile_sha256=profile_sha))
  return dict(summary=str(summary),objects=result),[summary,*external_assets,*[p for r in result for p in r['artifacts']]]
 def observations(self):
  output=self.run/'observations'
  if not(output/'OBSERVATIONS.json').exists():
   if output.exists():raise ValueError('Partial render preserved; new run needed')
   self.command('render',[sys.executable,'-B',str(BOYA/'render_collection_observations.py'),'--objects-json',str(self.run/'objects.json'),'--output-dir',str(output)])
  index=read(output/'OBSERVATIONS.json');assets=[output/'OBSERVATIONS.json']
  if index['status']!='complete'or len(index['observations'])!=50 or index['source_objects_sha256']!=sha(self.run/'objects.json'):raise ValueError('Incomplete or foreign render receipt')
  for row in index['observations']:
   p=Path(row['observation_json']);verify({str(p):row['sha256']});obs=read(p);assets.append(p)
   for x in obs['files'].values():verify({x['path']:x['sha256']});assets.append(Path(x['path']))
   if obs.get('explicit_physics_steps')!=0 or obs.get('robot_present'):raise ValueError('Observation scene boundary changed')
  return dict(objects=50,physics_steps=0),assets
 def representation(self):
  output=self.run/'representation_stage';batch=output/'BATCH_REPRESENTATION.json'
  if not batch.exists():
   if output.exists()and any(output.iterdir()):raise ValueError('Partial representation preserved')
   self.command('representation',[sys.executable,'-B',str(BOYA/'run_collection_representation.py'),'--manifest',str(self.run/'representation_manifest.json'),'--run-dir',str(output),'--device',self.c['representation_device'],'--checkpoint',self.c['checkpoint'],'--native-deps',self.c['native_deps'],'--seed',str(self.c['seed'])])
  d=read(batch)
  if d.get('manifest_sha256')!=sha(self.run/'representation_manifest.json')or d['status']!='all_representation_forwards_complete'or d['network_forward_count']!=50 or d['model_load_count']!=1 or not d['manifest_unchanged_at_end']:raise ValueError('Expected one shared CGR load and50 real forwards')
  assets=[batch]
  for row in d['objects']:
   p=Path(row['result_json']);verify({str(p):row['result_sha256']});f=read(p);assets.append(p)
   for item in f['output_files'].values():verify({item['path']:item['sha256']});assets.append(Path(item['path']))
  return dict(model_load_count=1,network_forward_count=50),assets
 def geometry(self):
  def one(job):
   out=Path(job['candidate_dir']);out.parent.mkdir(parents=True,exist_ok=True);generation=out/'generation.json';null=out/'no_candidate.json'
   if not generation.exists()and not null.exists():
    if out.exists():raise ValueError('Partial pool preserved: '+str(out))
    row=next(r for r in self.objects if r['object_id']==job['object_id'])
    self.command('pool_'+job['trial_id'],[sys.executable,'-B',str(BOYA/'build_collection_pool.py'),'--observation',job['observation_json'],'--frame-dir',row['representation_dir'],'--calibration',self.c['calibration'],'--gesture-id',str(job['gesture_id']),'--extra-depth-index',str(job['extra_depth_index']),'--candidate-limit',str(self.c['candidate_limit']),'--keep',str(self.c['keep']),'--width-expansion','1.7','--opponent','ff','--output-dir',str(out)],allowed_failure=True)
   if generation.exists():
    d=read(generation);verify(d['input_sha256']);verify({str(out/'candidates.npz'):d['candidates_sha256']})
    if not d['complete']or d['gesture_id']!=job['gesture_id']or d['additional_depth_index']!=job['extra_depth_index']:raise ValueError('Pool identity differs')
    return dict(trial_id=job['trial_id'],status='ready'),[generation,out/'candidates.npz',out/'world_object_pose.json']
   if null.exists():
    d=read(null);verify({job['source_npz']:d['source_candidates_sha256'],job['observation_json']:d['source_observation_sha256'],self.c['calibration']:d['calibration_sha256']})
    if d.get('physics_steps')!=0 or d.get('training_label')is not None:raise ValueError('Null pool acquired a fake label')
    return dict(trial_id=job['trial_id'],status='no_candidate',training_label=None),[null]
   raise RuntimeError('Pool failed without normal no_candidate receipt: '+job['trial_id'])
  rows=self.parallel(self.p['slots'],self.c['geometry_workers'],one);path=self.run/'POOL_PREPARATION.json';save(path,dict(complete=True,jobs=[r for r,_ in rows],new_network_forwards=0,new_physics_steps=0))
  return dict(ready=sum(r['status']=='ready'for r,_ in rows),no_candidate=sum(r['status']=='no_candidate'for r,_ in rows)),[path,*[p for _,files in rows for p in files]]
 def planning(self):
  from prepare_collection_queue import prepare_one
  native=read(self.run/'native_preprocessing/SUMMARY.json');repair={r['object_id']:r for r in native['results']};jobs=copy.deepcopy(self.p['slots'])
  for job in jobs:
   r=repair[job['object_id']]
   if r['repair_path']:job.update(collision_refinement=r['repair_path'],collision_refinement_sha256=r['repair_sha256'])
  path=self.run/'preparation_jobs.json';immutable(path,dict(backend_config=self.c['backend_config'],jobs=jobs));output=self.run/'planning';manifest=output/'queue_manifest.json'
  if not manifest.exists():
   if output.exists():raise ValueError('Partial planning preserved; no automatic rerun')
   output.mkdir();shutil.copyfile(path,output/'jobs.json')
   doc=dict(schema='anydex_boya_collection_queue_v1',complete=False,trials=[],nulls=[],errors=[],jobs_json=str(path),jobs_sha256=sha(path),source_sha256=hashes([BOYA/n for n in('prepare_collection_queue.py','trial_record.py','asset_adapter.py','native_arm.py')]),physical_steps=0,learned_Boya_head_loaded=False)
   save(manifest,doc)
   def one(job):
    verify(self.p['source_sha256']);value=prepare_one(job,output,Path(self.c['backend_config']));verify(self.p['source_sha256'])
    if value['status']not in('ready','no_hand_geometry_in_declared_budget','no_full_arm_plan_in_declared_budget'):raise ValueError('Unknown planning outcome')
    return dict(value,gesture_id=job['gesture_id'],extra_depth_index=job['extra_depth_index'],split=job['split'])
   def completed(job,value,bad):
    if bad:doc['errors'].append(dict(trial_id=job['trial_id'],error=value.get('error')))
    else:doc['trials'if value['status']=='ready'else'nulls'].append(value)
    save(manifest,doc)
   work=bounded_jobs(jobs,self.c['plan_workers'],one,completed);seen={r['item']['trial_id']for r in work};order={j['trial_id']:i for i,j in enumerate(jobs)}
   for key in('trials','nulls'):doc[key].sort(key=lambda r:order[r['trial_id']])
   doc.update(complete=not doc['errors']and len(work)==len(jobs),inputs_unchanged=sha(path)==doc['jobs_sha256'],not_dispatched_trial_ids=[j['trial_id']for j in jobs if j['trial_id']not in seen]);save(manifest,doc)
  d=read(manifest)
  if not d['complete']or d['errors']or not d['inputs_unchanged']or d['jobs_sha256']!=sha(path):raise ValueError('Planning infrastructure error; no physical dispatch')
  verify(d['source_sha256']);assets=[path,manifest,output/'jobs.json'];all_ids=[r['trial_id']for r in d['trials']+d['nulls']]
  if len(all_ids)!=self.p['slot_count']or len(set(all_ids))!=len(all_ids)or set(all_ids)!={j['trial_id']for j in jobs}:raise ValueError('Queue lost/repeated a declared slot')
  for row in d['trials']:
   verify(row['input_sha256']);plan=Path(row['plan_dir']);verify({str(plan/'commands.npz'):row['commands_sha256'],row['binding_json']:row['binding_sha256']});assets.extend([plan/n for n in('plan.json','commands.npz','path_geometry.json','lift_arm_geometry.json')]);assets.extend([Path(row['binding_json']),plan.parent/'preparation.json',plan.parent/'arm_collision_refinements.json'])
  for row in d['nulls']:assets.append(output/row['trial_id']/'preparation.json')
  return dict(manifest=str(manifest),admitted=len(d['trials']),nulls=len(d['nulls']),new_physics_steps=0),assets
 def execution(self):
  manifest=self.run/'planning/queue_manifest.json';queue=read(manifest);output=self.run/'execution'
  if output.exists():raise ValueError('Existing physical segment retained; only a completed stage receipt may reuse it; never automatically replay')
  output.mkdir();(output/'object_queues').mkdir();started=time.monotonic();assets=[manifest];jobs=[]
  for obj in self.objects:
   oid=obj['object_id'];trials=[r for r in queue['trials']if r['object_id']==oid];nulls=[r for r in queue['nulls']if r['object_id']==oid]
   if len(trials)>self.p['per_object_slot_limit']or len(trials)>16:raise ValueError('Object queue exceeds frozen resident plan limit')
   sub=dict(queue,trials=trials,nulls=nulls,parent_queue_manifest=str(manifest),parent_queue_manifest_sha256=sha(manifest))
   path=output/'object_queues'/(oid+'.json');immutable(path,sub);assets.append(path);jobs.append(dict(object_id=oid,manifest=str(path),manifest_sha256=sha(path),ready=len(trials),nulls=len(nulls)))
  progress=dict(schema='anydex_boya_object_execution_v2',status='running',protocol_sha256=sha(self.run/'protocol.json'),execution_workers=self.c['execution_workers'],objects={},peak_inflight_object_jobs=0,dispatch_events=[])
  save(output/'PROGRESS.json',progress)
  def one(job):
   oid=job['object_id'];sub=read(job['manifest']);verify({job['manifest']:job['manifest_sha256']})
   if not job['ready']:return dict(object_id=oid,status='no_ready_plans',result=dict(status='complete_with_null',trials=[],preparation_nulls=sub['nulls'],actual_grasps=0,physical_steps=0,positives=0,negatives=0,nulls=len(sub['nulls']),input_sha256_unchanged=True,source_sha256_unchanged=True))
   target=output/oid;proc=None;error=None
   try:proc=self.command('execution_'+oid,[sys.executable,'-B',str(BOYA/'execute_collection_queue.py'),'--manifest',job['manifest'],'--output-dir',str(target),'--device','cpu','--mode','serial','--batch-size',str(self.c['batch_size'])],allowed_failure=True)
   except Exception:error=traceback.format_exc()
   doc=read(target/'RESULTS.json')if(target/'RESULTS.json').exists()else{}
   ok=execution_usable(doc,[r['trial_id']for r in sub['trials']],proc['returncode']if proc else 1)
   return dict(object_id=oid,status='complete'if ok else'infrastructure_error',infrastructure_error=not ok,error=error,result=doc,process=proc,output_dir=str(target))
  def completed(job,value,bad):
   progress['objects'][job['object_id']]=value;progress['wall_s']=time.monotonic()-started
   save(output/'PROGRESS.json',progress)
  def dispatched(job,active):
   count=sum(bool(j['ready'])for j in active);progress['peak_inflight_object_jobs']=max(progress['peak_inflight_object_jobs'],count)
   progress['dispatch_events'].append(dict(object_id=job['object_id'],ready=job['ready'],inflight_ready_object_jobs=count,elapsed_wall_s=time.monotonic()-started));save(output/'PROGRESS.json',progress)
  work=bounded_jobs(jobs,self.c['execution_workers'],one,completed,dispatched);failed=any(r['failed']for r in work)
  for job in jobs:
   if job['object_id']not in progress['objects']:progress['objects'][job['object_id']]=dict(object_id=job['object_id'],status='not_dispatched_after_infrastructure_error',result={})
  objects=[progress['objects'][job['object_id']]for job in jobs];rows=[r for obj in objects for r in obj.get('result',{}).get('trials',[])];byid={j['trial_id']:j for j in self.p['slots']};records=[]
  for obj in objects:
   if not obj.get('output_dir'):continue
   target=Path(obj['output_dir'])
   for name in('RESULTS.json','PREPARED.json','training_records.jsonl'):
    p=target/name
    if p.exists():assets.append(p)
   for row in obj.get('result',{}).get('trials',[]):
    record=target/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
    if record.exists():
     assets.append(record);value=read(record)
     if value.get('label_eligible')is True and value.get('result')in(0,1):
      slot=byid[row['trial_id']];value['collection_context']=dict(trial_id=slot['trial_id'],object_id=slot['object_id'],split=slot['split'],gesture_id=slot['gesture_id'],extra_depth_index=slot['extra_depth_index'],benchmark_only=False,protocol_sha256=sha(self.run/'protocol.json'));records.append(value)
    for name in('RESULTS.json','bound_evaluation.json'):
     p=record.parent/name
     if p.exists():assets.append(p)
    native=Path(row['execution_dir'])/'evaluation.json'if row.get('execution_dir')else None
    if native is not None and native.exists():
     e=read(native);assets.append(native);trace=native.parent/'trace.npz'
     if trace.exists():verify({str(trace):e['trace_sha256']});assets.append(trace)
  counts=label_counts(self.p,rows);costs=execution_costs(rows,[r['trial_id']for r in queue['trials']]);unexecuted=costs['not_dispatched_ready_trial_ids'];grouped=split_records(records)
  if not failed and(unexecuted or len(records)!=len(queue['trials'])):failed=True
  result=dict(schema='anydex_boya_collection_execution_v2',status='infrastructure_stopped'if failed else'complete_with_null'if queue['nulls']else'complete',complete=not failed,protocol_sha256=sha(self.run/'protocol.json'),trials=rows,object_results=objects,preparation_nulls=queue['nulls'],unexecuted_ready_trial_ids=unexecuted,unknown_execution_trial_ids=costs['unknown_execution_trial_ids'],actual_grasps=costs['actual_grasps'],physical_steps=costs['physical_steps'],positives=sum(r.get('label')==1 for r in rows),negatives=sum(r.get('label')==0 for r in rows),nulls=len(queue['nulls'])+len({r['trial_id']for r in rows if r.get('label')is None}|set(unexecuted)),wall_s=time.monotonic()-started,execution_workers=self.c['execution_workers'],input_sha256_unchanged=True,source_sha256_unchanged=True,cross_object_residency=False,serial_resident_within_object_wave=True)
  verify(self.p['input_sha256']);verify(self.p['source_sha256']);verify(self.p['manifest_sha256'])
  resultfile=output/'RESULTS.json';save(resultfile,result)
  exports={'all_records.jsonl':records,'train_records.jsonl':grouped['train'],'validation_records.jsonl':grouped['validation']}
  for filename,values in exports.items():
   with(output/filename).open('x')as f:
    for value in values:f.write(json.dumps(value,allow_nan=False)+'\n')
  summary=self.run/'collection_counts.json';save(summary,dict(slot_count=self.p['slot_count'],split_slot_counts=self.p['split_slot_counts'],gesture_ids=self.p['gesture_ids'],valid_label_target_totals=self.p['valid_label_target_totals'],targets_are_valid_labels_not_positive_labels=True,counts=counts,actual_grasps=result['actual_grasps'],physical_steps=result['physical_steps'],nulls=result['nulls'],unexecuted_ready_trial_ids=unexecuted,all_extra_records_retained=True,head_trained=False,complete=not failed))
  progress.update(status=result['status'],wall_s=result['wall_s']);save(output/'PROGRESS.json',progress)
  assets.extend([resultfile,*[output/name for name in exports],output/'PROGRESS.json',summary])
  if failed:raise RuntimeError('Physical infrastructure error; finite inflight objects drained, no automatic replay; partial outputs retained')
  return dict(actual_grasps=result['actual_grasps'],physical_steps=result['physical_steps'],labels=len(records),nulls=result['nulls'],counts=str(summary),execution_wall_s=result['wall_s']),assets


def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--phase',choices=('prepare','execute','all'),default='prepare');p.add_argument('--dry-run',action='store_true',help='Freeze dynamic slot manifests only; zero Kit/model/physics calls')
 p.add_argument('--gesture-ids',nargs='+',type=int,default=None,help='One to four distinct calibrated gesture IDs; default 1 6 7 8')
 for key,value in DEFAULTS.items():
  if key!='gesture_ids':p.add_argument('--'+key.replace('_','-'),type=int if isinstance(value,int)else str,default=None)
 a=p.parse_args()
 for name,lo,hi in(('geometry_workers',1,2),('execution_workers',1,2),('plan_workers',1,8),('batch_size',1,16),('candidate_limit',1,100000),('keep',1,1000),('max_arm_plans',1,32)):
  if getattr(a,name)is not None and not lo<=getattr(a,name)<=hi:p.error('Invalid '+name)
 if a.physics_device is not None and a.physics_device!='cpu':p.error('v2 physical object queues use cpu')
 if a.execution_mode is not None and a.execution_mode!='serial':p.error('v2 physical object queues use serial residency')
 if a.gesture_ids is not None:
  try:validate_gestures(a.gesture_ids)
  except ValueError as e:p.error(str(e))
 return a


def main():
 a=parse_args();run=a.run_dir.resolve();run.parent.mkdir(parents=True,exist_ok=True)
 with(run.parent/('.'+run.name+'.pipeline.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=freeze(run,a);pipeline=Pipeline(run,p);pipeline.check()
  if a.dry_run:
   immutable(run/'DRY_PLAN.json',dict(protocol_sha256=sha(run/'protocol.json'),slot_count=p['slot_count'],train_slots=p['split_slot_counts']['train'],validation_slots=p['split_slot_counts']['validation'],gesture_ids=p['gesture_ids'],valid_label_target_totals=p['valid_label_target_totals'],native_assets=50,rendered_frames_planned=50,CGR_model_loads_planned=1,CGR_forwards_planned=50,geometry_workers=p['config']['geometry_workers'],CPU_plan_workers=p['config']['plan_workers'],CPU_execution_workers=p['config']['execution_workers'],native_preflight=p['config']['native_preflight'],per_object_slot_limit=p['per_object_slot_limit'],physics_device=p['config']['physics_device'],physics_mode=p['config']['execution_mode'],batch_size=p['config']['batch_size'],new_model_loads=0,new_network_forwards=0,new_physics_steps=0));pipeline.publish('frozen_dry_plan_only');print(json.dumps(read(run/'DRY_PLAN.json'),indent=2));return
  try:
   functions=[pipeline.native,pipeline.observations,pipeline.representation,pipeline.geometry,pipeline.planning]
   if a.phase=='execute':
    for name in STAGES[:-1]:
     f=run/'stages'/(name+'.json')
     if not f.exists()or read(f).get('complete')is not True or read(f).get('protocol_sha256')!=sha(run/'protocol.json'):raise ValueError('Explicit execute requires completed prepare: '+name)
     verify(read(f)['artifacts_sha256'])
   else:
    for name,fn in zip(STAGES,functions):pipeline.stage(name,fn)
   if a.phase in('execute','all'):
    pipeline.progress['physics_steps']=None;result=pipeline.stage('execution',pipeline.execution);pipeline.progress['physics_steps']=result.get('physical_steps',0);pipeline.publish('collection_complete')
   else:pipeline.publish('preparation_complete_execution_not_requested_this_invocation')
  except BaseException:
   pipeline.progress['error']=traceback.format_exc();pipeline.publish('infrastructure_stopped');raise

if __name__=='__main__':main()
