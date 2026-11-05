"""Stage the frozen 40/10 AnyDex-Boya collection; prepare never executes grasps.

--dry-run freezes the input/command plan only: no Kit, model import or physics.
Explicit --phase execute/all enables the original source-bound physical queue.
"""
from __future__ import annotations
import argparse,collections,copy,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path

ROOT=Path('/home/wangrenpeng/AnyDexGrasp');BOYA=ROOT/'reproduction/boya'
NATIVE=Path('/home/wangrenpeng/CanonDex/scripts/research/preprocess_native_object.py')
DEFAULTS=dict(selection=str(ROOT/'outputs/collection40_20261105_v1/selection_v2/object_selection.json'),calibration=str(ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json'),backend_config=str(ROOT/'outputs/boya_pilot8_20261105_v1/backend_pilot.json'),checkpoint='/mnt/public/datasets/AnyDexGrasp/logs/model/checkpoint.tar.18',native_deps=str(ROOT/'outputs/native_deps'),representation_device='cuda:0',physics_device='cpu',execution_mode='serial',batch_size=8,geometry_workers=2,plan_workers=4,candidate_limit=500,keep=8,max_arm_plans=4,seed=20261105)
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


def freeze(run,args):
 old=read(run/'protocol.json')if(run/'protocol.json').exists()else None
 cfg={k:getattr(args,k)if getattr(args,k)is not None else(old['config'][k]if old else v)for k,v in DEFAULTS.items()}
 for key in('selection','calibration','backend_config','checkpoint','native_deps'):cfg[key]=str(Path(cfg[key]).resolve())
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
 backend=read(cfg['backend_config']);cal=read(cfg['calibration'])
 for v in backend.values():
  if isinstance(v,str)and v.startswith('/')and Path(v).is_file():add(v)
 for p in Path(backend['reference_root']).rglob('*.json'):add(p)
 for p,meta in cal['source_lock']['source_files'].items():add(p,meta['sha256'])
 for p,h in cal['source_lock']['readonly_fk_dependencies'].items():add(p,h)
 for g in(1,6,8):
  if not any(s['gesture_id']==g and s['self_collision_clear']and s['opposing_group']=='ff'for s in cal['slots']):raise ValueError('Missing clear calibration datum for gesture '+str(g))
 # Only source assets/control enter the freeze, never actual grasp outcomes.
 for row in rows:
  for pathkey,hashkey in(('mesh_path','mesh_sha256'),('urdf_path','urdf_sha256'),('pose_path','pose_sha256'),('metadata_path','metadata_sha256')):add(row[pathkey],row[hashkey])
  for part in row['convex_parts']:add(part['path'],part['sha256'])
 source_names=['render_collection_observations.py','render_pilot_observations.py','run_collection_representation.py','run_pilot_representation.py','build_collection_pool.py','prepare_collection_queue.py','execute_collection_queue.py','asset_adapter.py','trial_record.py','native_arm.py']
 sources=[Path(__file__).resolve(),*(BOYA/n for n in source_names),NATIVE,ROOT/'reproduction/runtime/run_official_rgbd_frame.py',ROOT/'robot_dh3.py',ROOT/'models/minkowski_graspnet_single_point.py',ROOT/'utils/pt_utils.py']
 canon=NATIVE.parents[2]
 sources.extend(canon/n for n in('canondex/grasp/native_object_preprocess.py','canondex/grasp/collision_refinement.py','canondex/grasp/convex_partition.py','canondex/grasp/physics.py','canondex/scene/convex_cooking.py'))
 rebased=copy.deepcopy(selection)
 for row in rebased['objects']:
  row.update(id=row['object_id'],observation_json=str(run/'observations'/row['object_id']/'observation.json'),representation_dir=str(run/'representations'/row['object_id']))
  row['observation']=row['observation_json'];row['pools']={str(g):{str(d):str(run/'pools'/row['object_id']/('g'+str(g))/('d'+str(d)))for d in range(4)}for g in(1,6,8)}
 immutable(run/'objects.json',rebased)
 immutable(run/'representation_manifest.json',dict(objects=[{k:r[k]for k in('object_id','observation_json','representation_dir','split')}for r in rebased['objects']]))
 slots=[dict(trial_id=r['object_id']+'__g%02d__d%d'%(g,d),object_id=r['object_id'],split=r['split'],gesture_id=g,extra_depth_index=d,candidate_dir=r['pools'][str(g)][str(d)],observation_json=r['observation_json'],source_npz=str(Path(r['representation_dir'])/'raw_camera_candidates.npz'),calibration=cfg['calibration'],max_arm_plans=cfg['max_arm_plans'])for r in rebased['objects']for g in(1,6,8)for d in range(4)]
 protocol=dict(schema='anydex_boya_collection_pipeline_v1',config=cfg,selection_sha256=files[cfg['selection']],input_sha256=files,source_sha256=hashes(sources),manifest_sha256=hashes([run/'objects.json',run/'representation_manifest.json']),selection_pool_contract_inherited=selection.get('pool_contract'),pool_contract_amendment=dict(reason='Explicit collection expansion authorized after pilot; original selector declared only inherited extra depth0. This pipeline alone declares four depths before any stage.',gesture_ids=[1,6,8],extra_depth_indices=[0,1,2,3],extra_depth_m=[0.,.01,.02,.03]),gesture_ids=[1,6,8],extra_depth_indices=[0,1,2,3],extra_depth_m=[0.,.01,.02,.03],slots=slots,slot_count=600,split_slot_counts={'train':480,'validation':120},valid_label_targets_per_gesture={'train':100,'validation':20},native_asset_policy=dict(max_failed_partitions=32,physics_steps=0,source_union_preserved=True,unchanged_native_tolerance_m=20e-6),scope='Declared3collection gestures, not a best-gesture claim; one RGB-D and one shared original identity CGR forward per object. One first arm-admitted actual trial per gesture/depth slot, finite source score order, no outcome feedback or invented labels. Validation is adaptation-development heldout, not unseen-CGR-pretraining.',created_unix_s=time.time())
 save(run/'protocol.json',protocol);return protocol


class Pipeline:
 def __init__(self,run,protocol):
  self.run=run;self.p=protocol;self.c=protocol['config'];self.objects=read(run/'objects.json')['objects'];self.log=run/'logs';self.log.mkdir(exist_ok=True);(run/'stages').mkdir(exist_ok=True)
  self.progress=read(run/'pipeline_progress.json')if(run/'pipeline_progress.json').exists()else dict(schema='anydex_boya_pipeline_progress_v1',stages={},slot_count=600,physics_steps=0)
 def check(self):verify(self.p['input_sha256']);verify(self.p['source_sha256']);verify(self.p['manifest_sha256'])
 def publish(self,status):self.progress.update(status=status,updated_unix_s=time.time());save(self.run/'pipeline_progress.json',self.progress)
 def command(self,tag,argv,allowed_failure=False):
  verify(self.p['source_sha256']);start=time.monotonic()
  with(self.log/(tag+'.log')).open('xb')as log:r=subprocess.run(argv,cwd=ROOT,env=env(),stdout=log,stderr=subprocess.STDOUT)
  verify(self.p['source_sha256'])
  if r.returncode and not allowed_failure:raise RuntimeError('Infrastructure failure '+tag+'; logs/outputs retained')
  return dict(argv=argv,returncode=r.returncode,wall_s=time.monotonic()-start)
 def parallel(self,items,workers,fn):
  # On infrastructure error stop new dispatch; already running jobs finish.
  items=iter(items);result=[];error=None
  with ThreadPoolExecutor(max_workers=workers)as ex:
   active={ex.submit(fn,x)for x in __import__('itertools').islice(items,workers)}
   while active:
    done,active=wait(active,return_when=FIRST_COMPLETED)
    for f in done:
     try:result.append(f.result())
     except Exception as e:error=error or e
    if error is None:
     for x in __import__('itertools').islice(items,len(done)):active.add(ex.submit(fn,x))
  if error:raise error
  return result
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
  def one(row):
   target=output/row['object_id']
   argv=[sys.executable,'-B',str(NATIVE),'--object-root',row['object_root'],'--scale',str(row['scale']),'--physics-config',read(self.c['backend_config'])['physics_config'],'--cache-dir',str(self.run/'native_object_cache'),'--output-dir',str(target),'--max-failed-partitions','32']
   resultfile=target/'result.json'
   if not resultfile.exists():
    if target.exists():raise ValueError('Partial native preprocessing preserved: '+str(target))
    self.command('native_'+row['object_id'],argv)
   d=read(resultfile)
   if not(d.get('complete')and d.get('trial_physics_steps')==0 and d.get('original_or_refined_within_20um')):raise ValueError('Native original20um gate did not pass')
   bound=d['binding']['inputs']
   if bound['object_id']!=row['object_id']or bound['scale']!=row['scale']or bound['physics_profile_sha256']!=sha(read(self.c['backend_config'])['physics_config'])or bound['max_failed_partitions']!=32 or bound['tolerance_m']!=20e-6:raise ValueError('Native asset binding differs')
   for relative,h in bound['implementations'].items():verify({str(NATIVE.parents[2]/relative):h})
   for relative,h in bound['source_files'].items():verify({str(Path(row['object_root'])/relative):h})
   cache=Path(d['cache_directory']);assets=[resultfile]
   for relative,h in d.get('artifact_sha256',{}).items():verify({str(cache/relative):h});assets.append(cache/relative)
   repair=d.get('repair_path')
   if repair:assets.append(Path(repair))
   return dict(object_id=row['object_id'],result_json=str(resultfile),repair_path=repair,repair_sha256=sha(repair)if repair else None,artifacts=list(map(str,assets)))
  result=self.parallel(self.objects,2,one);summary=output/'SUMMARY.json';save(summary,dict(complete=True,results=result,max_failed_partitions=32,new_physics_steps=0))
  return dict(summary=str(summary),objects=result),[summary,*[p for r in result for p in r['artifacts']]]
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
  native=read(self.run/'native_preprocessing/SUMMARY.json');repair={r['object_id']:r for r in native['results']};jobs=copy.deepcopy(self.p['slots'])
  for job in jobs:
   r=repair[job['object_id']]
   if r['repair_path']:job.update(collision_refinement=r['repair_path'],collision_refinement_sha256=r['repair_sha256'])
  path=self.run/'preparation_jobs.json';immutable(path,dict(backend_config=self.c['backend_config'],jobs=jobs));output=self.run/'planning';manifest=output/'queue_manifest.json'
  if not manifest.exists():
   if output.exists():raise ValueError('Partial planning preserved; no automatic rerun')
   self.command('planning',[sys.executable,'-B',str(BOYA/'prepare_collection_queue.py'),'--jobs-json',str(path),'--output-dir',str(output),'--workers',str(self.c['plan_workers'])])
  d=read(manifest)
  if not d['complete']or d['errors']or not d['inputs_unchanged']or d['jobs_sha256']!=sha(path):raise ValueError('Planning infrastructure error; no physical dispatch')
  verify(d['source_sha256']);assets=[path,manifest,output/'jobs.json']
  if len(d['trials'])+len(d['nulls'])!=600:raise ValueError('Queue lost a declared slot')
  for row in d['trials']:
   verify(row['input_sha256']);plan=Path(row['plan_dir']);verify({str(plan/'commands.npz'):row['commands_sha256'],row['binding_json']:row['binding_sha256']});assets.extend([plan/'plan.json',plan/'commands.npz',Path(row['binding_json']),plan.parent/'preparation.json',plan.parent/'arm_collision_refinements.json'])
  for row in d['nulls']:assets.append(output/row['trial_id']/'preparation.json')
  return dict(manifest=str(manifest),admitted=len(d['trials']),nulls=len(d['nulls']),new_physics_steps=0),assets
 def execution(self):
  manifest=self.run/'planning/queue_manifest.json';queue=read(manifest);output=self.run/'execution';resultfile=output/'RESULTS.json'
  if not queue['trials']:
   output.mkdir(exist_ok=True);save(resultfile,dict(status='complete_with_null',trials=[],preparation_nulls=queue['nulls'],actual_grasps=0,physical_steps=0))
   counts={split:{str(g):dict(labels=0,positive=0,negative=0,target=100 if split=='train'else 20,shortfall=100 if split=='train'else 20,extra_valid_labels_retained=0)for g in(1,6,8)}for split in('train','validation')}
   summary=self.run/'collection_counts.json';save(summary,dict(slot_count=600,targets_are_valid_labels_not_positive_labels=True,counts=counts,actual_grasps=0,physical_steps=0,nulls=600,all_extra_records_retained=True,head_trained=False))
   return dict(actual_grasps=0,physical_steps=0,labels=0,nulls=600,counts=str(summary)),[resultfile,summary]
  if not resultfile.exists():
   self.command('execution',[sys.executable,'-B',str(BOYA/'execute_collection_queue.py'),'--manifest',str(manifest),'--output-dir',str(output),'--device',self.c['physics_device'],'--mode',self.c['execution_mode'],'--batch-size',str(self.c['batch_size'])])
  d=read(resultfile)
  if any(r.get('label')is None for r in d.get('trials',[]))or d.get('infrastructure_error')or d.get('error')or d['status']not in('complete','complete_with_null')or not d.get('input_sha256_unchanged')or not d.get('source_sha256_unchanged'):raise ValueError('Physical infrastructure error; retained outputs may never be automatically replayed')
  assets=[resultfile,output/'PREPARED.json',output/'training_records.jsonl'];byid={j['trial_id']:j for j in self.p['slots']};counts={s:{str(g):dict(labels=0,positive=0,negative=0,target=100 if s=='train'else 20)for g in(1,6,8)}for s in('train','validation')}
  for row in d['trials']:
   job=byid[row['trial_id']]
   if row['label']in(0,1)and row['label_eligible']:
    c=counts[job['split']][str(job['gesture_id'])];c['labels']+=1;c['positive'if row['label']==1 else'negative']+=1
   record=output/'trials'/row.get('output_identity',row['trial_id'])/'training_record.json'
   if record.exists():assets.append(record)
   for name in('RESULTS.json','bound_evaluation.json'):
    path=record.parent/name
    if path.exists():assets.append(path)
   native=Path(row['execution_dir'])/'evaluation.json';e=read(native);trace=native.parent/'trace.npz';verify({str(trace):e['trace_sha256']});assets.extend([native,trace])
  for split in counts:
   for count in counts[split].values():count['shortfall']=max(0,count['target']-count['labels']);count['extra_valid_labels_retained']=max(0,count['labels']-count['target'])
  summary=self.run/'collection_counts.json';save(summary,dict(slot_count=600,targets_are_valid_labels_not_positive_labels=True,counts=counts,actual_grasps=d['actual_grasps'],physical_steps=d['physical_steps'],nulls=d['nulls'],all_extra_records_retained=True,head_trained=False))
  return dict(actual_grasps=d['actual_grasps'],physical_steps=d['physical_steps'],labels=d['positives']+d['negatives'],nulls=d['nulls'],counts=str(summary)),[*assets,summary]


def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--phase',choices=('prepare','execute','all'),default='prepare');p.add_argument('--dry-run',action='store_true',help='Freeze manifests and600-slot command plan only; zero Kit/model/physics calls')
 for key,value in DEFAULTS.items():p.add_argument('--'+key.replace('_','-'),type=int if isinstance(value,int)else str,default=None)
 a=p.parse_args()
 for name,lo,hi in(('geometry_workers',1,2),('plan_workers',1,8),('batch_size',1,16),('candidate_limit',1,100000),('keep',1,1000),('max_arm_plans',1,32)):
  if getattr(a,name)is not None and not lo<=getattr(a,name)<=hi:p.error('Invalid '+name)
 if a.physics_device is not None and a.physics_device not in('cpu','cuda:0'):p.error('physics-device cpu/cuda:0 required')
 if a.execution_mode is not None and a.execution_mode not in('serial','parallel'):p.error('execution-mode serial/parallel required')
 return a


def main():
 a=parse_args();run=a.run_dir.resolve();run.parent.mkdir(parents=True,exist_ok=True)
 with(run.parent/('.'+run.name+'.pipeline.lock')).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=freeze(run,a);pipeline=Pipeline(run,p);pipeline.check()
  if a.dry_run:
   immutable(run/'DRY_PLAN.json',dict(protocol_sha256=sha(run/'protocol.json'),slot_count=600,train_slots=480,validation_slots=120,native_assets=50,rendered_frames_planned=50,CGR_model_loads_planned=1,CGR_forwards_planned=50,geometry_workers=p['config']['geometry_workers'],CPU_plan_workers=p['config']['plan_workers'],physics_device=p['config']['physics_device'],physics_mode=p['config']['execution_mode'],batch_size=p['config']['batch_size'],new_model_loads=0,new_network_forwards=0,new_physics_steps=0));pipeline.publish('frozen_dry_plan_only');print(json.dumps(read(run/'DRY_PLAN.json'),indent=2));return
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
