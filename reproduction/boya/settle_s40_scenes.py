"""Settle source S40 objects on the native table before fresh RGB-D capture.

Original object spawn/physics are reused. An AST overlay changes only object
prim namespaces to permit independent lanes; CanonDex files remain unchanged.
Trials later cold-start at the measured pose with explicitly declared zero
velocity. This is initialization, not continuous live-scene regrasp.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');CANON=Path('/home/wangrenpeng/CanonDex')
POLICY=dict(stability_basis='Continuous measured pose stationarity, native twist sanity, then explicit cold-recreation hold; instantaneous solver twist is retained',dt_s=.004,min_steps=500,max_steps=5000,window_steps=250,max_linear_speed_m_s=.02,
 max_angular_speed_rad_s=.5,max_window_translation_m=.0002,max_window_rotation_rad=.005,
 cold_recreation_steps=750,max_recreation_translation_m=.001,max_recreation_rotation_rad=.005,lane_spacing_m=2.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def verify(files):
 for p,h in files.items():
  if sha(p)!=h:raise ValueError('Frozen input changed: '+p)
def window_metrics(poses,velocities,policy=POLICY):
 p=np.asarray(poses,float);v=np.asarray(velocities,float);w=policy['window_steps']
 if p.ndim!=2 or p.shape[1]!=7 or v.shape!=(len(p),6)or not np.isfinite(p).all()or not np.isfinite(v).all():raise ValueError('Finite poses7 and velocities6 required')
 if len(p)<w:return dict(stable=False,reason='window_incomplete')
 p=p[-w:];v=v[-w:];q=p[:,3:];norm=np.linalg.norm(q,axis=1)
 if np.max(abs(norm-1))>1e-4:raise ValueError('Unit source XYZW quaternion required')
 q=q/norm[:,None];angles=2*np.arccos(np.clip(abs(q@q[-1]),0,1))
 result=dict(max_linear_speed_m_s=float(np.linalg.norm(v[:,:3],axis=1).max()),max_angular_speed_rad_s=float(np.linalg.norm(v[:,3:],axis=1).max()),
  max_window_translation_m=float(np.linalg.norm(p[:,:3]-p[-1,:3],axis=1).max()),max_window_rotation_rad=float(angles.max()))
 result['stable']=all(result[k]<=policy[k]for k in result);return result

def namespaced_spawn(module,path):
 """Copy exactly two original functions; substitute only fixed prim paths."""
 nodes=[copy.deepcopy(n)for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name in ['spawn_mesh_object','spawn_object']]
 assert len(nodes)==2
 class Paths(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str)and node.value.startswith('/World/Object'):
    return ast.copy_location(ast.Constant(path+node.value[len('/World/Object'):]),node)
   return node
 tree=ast.fix_missing_locations(Paths().visit(ast.Module(body=nodes,type_ignores=[])));ns=dict(vars(module));exec(compile(tree,str(module.__file__),'exec'),ns);return ns['spawn_object']

def worker(spec_path):
 spec=read(spec_path);out=Path(spec['output']);out.mkdir(parents=True,exist_ok=False);save(out/'STARTED.json',dict(pid=os.getpid(),physics_steps=0));verify(spec['inputs']);verify(spec['sources'])
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(headless=True,device='cpu',visualizer=None,kit_args='--/log/file=/dev/null');code=1;steps=0;started=time.monotonic()
 try:
  sys.path[:0]=[str(CANON),str(CANON/'scripts/research')]
  import torch,warp as wp
  import isaaclab.sim as sim_utils
  from pxr import UsdGeom
  from scipy.spatial.transform import Rotation
  from run_template_arm_positive import object_asset
  from canondex.grasp.physics import load_physics_config
  from canondex.scene.floor import spawn_floor_and_light,floor_collision_settings
  from canondex.scene.runtime import forward_scene
  from canondex.scene import grasp_object
  from canondex.grasp.arm_resident_scene import restore_declared_object_state
  from template_gripper_native_audit import audit_scene
  profile=load_physics_config(spec['physics_config']);dt=POLICY['dt_s']
  sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(device='cpu',dt=dt,gravity=(0.,0.,0.),physics=profile.scene_cfg(),use_fabric=False,visualizer_cfgs=[]))
  spawn_floor_and_light();UsdGeom.Xform.Define(sim.stage,'/World/Robot');UsdGeom.Xform.Define(sim.stage,'/World/Settle')
  bodies=[];origins=[];roots=[];initial=[]
  for i,row in enumerate(spec['objects']):
   origin=np.array([(i%5)*POLICY['lane_spacing_m'],(i//5)*POLICY['lane_spacing_m'],0.]);T=np.asarray(row['world_T_object']).copy();T[:3,3]+=origin
   asset=object_asset(SimpleNamespace(object_root=Path(row['object_root'])),T,row['scale']);path='/World/Settle/'+row['object_id'];repair=spec['native'][row['object_id']].get('repair_path')
   obj=namespaced_spawn(grasp_object,path)(sim,asset,physics_profile=profile,collision_refinement=read(repair)if repair else None)
   bodies.append(obj);origins.append(origin);roots.append(path);initial.append(asset.pose_xyzw)
  sim_utils.bind_physics_material('/World/ground',roots[0]+'Material');sim.reset()
  for obj,pose in zip(bodies,initial):
   obj.update(dt);restore_declared_object_state(sim,obj,pose)
   values=obj.root_view.get_material_properties().numpy().copy();values[...,:2]=.6;values[...,2]=0.
   ids=wp.array(np.arange(len(values),dtype=np.int32),dtype=wp.int32,device='cpu');obj.root_view.set_material_properties(wp.array(values,dtype=wp.float32,device='cpu'),ids)
   actual=obj.root_view.get_material_properties().numpy()
   if not np.allclose(actual,values,rtol=0,atol=1e-7):raise ValueError('Native object material changed')
   if float(obj.root_view.get_masses().numpy().ravel()[0])!=float(np.float32(.1)):raise ValueError('Native mass changed')
   obj.permanent_wrench_composer.reset();obj.permanent_wrench_composer.add_forces_and_torques_index(forces=torch.tensor([[[0.,0.,-.98]]],dtype=torch.float32,device=obj.device),is_global=True)
  (out/'native').mkdir()
  native=audit_scene(sim,out/'native',['all'],roots=tuple(roots),fast=True)
  if not native['within_20um']:raise ValueError('Original native20um object gate rejected')
  poses=[[]for _ in bodies];velocities=[[]for _ in bodies]
  def observe():
   for i,obj in enumerate(bodies):
    p=obj.data.root_link_pose_w.torch[0].cpu().numpy().copy();p[:3]-=origins[i];v=obj.data.root_link_vel_w.torch[0].cpu().numpy().copy();poses[i].append(p);velocities[i].append(v)
  def advance():
   nonlocal steps
   for obj in bodies:obj.write_data_to_sim()
   before=sim.get_physics_step_count();sim.step(render=False);steps+=1
   if sim.get_physics_step_count()!=before+1:raise RuntimeError('Unexpected physics clock')
   for obj in bodies:obj.update(dt)
   observe()
  observe();status=[]
  for count in range(1,POLICY['max_steps']+1):
   advance()
   if count>=POLICY['min_steps']and count%25==0:
    status=[window_metrics(p,v)for p,v in zip(poses,velocities)]
    if all(x['stable']for x in status):break
  settle_steps=steps;settled=[p[-1].astype(float).copy()for p in poses];stable=[window_metrics(p,v)for p,v in zip(poses,velocities)]
  for p in settled:p[3:]/=np.linalg.norm(p[3:])
  # Explicit independent-trial cold recreation; never claim continuous state.
  if all(x['stable']for x in stable):
   for i,(obj,p)in enumerate(zip(bodies,settled)):
    physical=p.copy();physical[:3]+=origins[i];restore_declared_object_state(sim,obj,physical)
    obj.permanent_wrench_composer.add_forces_and_torques_index(forces=torch.tensor([[[0.,0.,-.98]]],dtype=torch.float32,device=obj.device),is_global=True)
   for _ in range(POLICY['cold_recreation_steps']):advance()
  results=[]
  for i,row in enumerate(spec['objects']):
   p=np.asarray(poses[i]);v=np.asarray(velocities[i]);verify_p=p[settle_steps+1:];recreated=bool(len(verify_p)==POLICY['cold_recreation_steps']);shift=angle=None
   if recreated:
    shift=float(np.linalg.norm(verify_p[:,:3]-settled[i][:3],axis=1).max());q=verify_p[:,3:]/np.linalg.norm(verify_p[:,3:],axis=1)[:,None];ref=settled[i][3:]/np.linalg.norm(settled[i][3:]);angle=float((2*np.arccos(np.clip(abs(q@ref),0,1))).max())
   ready=bool(stable[i]['stable']and recreated and shift<=POLICY['max_recreation_translation_m']and angle<=POLICY['max_recreation_rotation_rad']and window_metrics(p,v)['stable'])
   target=out/row['object_id'];target.mkdir();trace=target/'settling_trace.npz';np.savez_compressed(trace,pose_xyzw=p,velocity=v,phase=np.asarray(['initial']+['settling']*settle_steps+['cold_recreation_check']*(steps-settle_steps)),step=np.arange(steps+1))
   T=np.eye(4);T[:3,:3]=Rotation.from_quat(settled[i][3:]).as_matrix();T[:3,3]=settled[i][:3]
   record=dict(object_id=row['object_id'],ready=ready,world_T_object=T.tolist(),measured_settled_pose_xyzw=settled[i].tolist(),measured_settled_velocity=v[settle_steps].tolist(),settle_window=stable[i],final_window=window_metrics(p,v),cold_recreation_verified=recreated,cold_recreation_max_translation_m=shift,cold_recreation_max_rotation_rad=angle,
    trace=str(trace),trace_sha256=sha(trace),settle_steps=settle_steps,recreation_steps=steps-settle_steps,physics_steps=steps,environment_origin_m=origins[i].tolist(),state_write_boundary='Only declared initial placement and one explicit cold-recreation test at the measured pose/zero velocity; no state writes during either integration segment')
   save(target/'RESULTS.json',record);results.append(record)
  verify(spec['inputs']);verify(spec['sources']);save(out/'RESULTS.json',dict(status='complete',objects=results,policy=POLICY,physics_ticks=steps,object_steps=len(bodies)*steps,wall_s=time.monotonic()-started,native_within_20um=True,native_hulls=native['hulls'],floor=floor_collision_settings(sim.stage),robot_present=False,grasp_attempts=0));code=0
 except BaseException:
  save(out/'ERROR.json',dict(error=traceback.format_exc(),physics_ticks=steps,object_steps=steps*len(spec['objects']),wall_s=time.monotonic()-started));raise
 finally:launcher.app.close(exit_code=code)

def run(args):
 out=args.output_dir.resolve()
 if out.exists():raise ValueError('New settling output required; never replay existing worker outputs')
 out.mkdir(parents=True);scene=read(args.scene_manifest);old=read(args.previous_collection/'protocol.json');native=read(args.previous_collection/'native_preprocessing/SUMMARY.json')
 if scene['schema']!='anydex_boya_s40_scene_manifest_v1'or len(scene['objects'])!=40:raise ValueError('All40 source S40 objects required')
 if any(r['split']!='train'or r['scale']!=1 for r in scene['objects']):raise ValueError('Original-scale S40 train only')
 backend=read(old['config']['backend_config']);inputs={str(args.scene_manifest.resolve()):sha(args.scene_manifest),str(args.previous_collection/'protocol.json'):sha(args.previous_collection/'protocol.json'),str(args.previous_collection/'native_preprocessing/SUMMARY.json'):sha(args.previous_collection/'native_preprocessing/SUMMARY.json')}
 for p,h in old['input_sha256'].items():inputs[p]=h
 for row in native['results']:
  for p in row['artifacts']:inputs[p]=sha(p)
 sources={str(Path(__file__).resolve()):sha(__file__)}
 for p in sorted([*(CANON/'canondex').rglob('*.py'),*(CANON/'scripts/research').glob('*.py')]):sources[str(p)]=sha(p)
 verify(inputs);objects=scene['objects'];size=10;specs=[]
 for i in range(0,len(objects),size):
  p=out/f'group_{i//size:02d}.WORK.json';spec=dict(objects=objects[i:i+size],output=str(out/f'group_{i//size:02d}'),native={r['object_id']:r for r in native['results']},physics_config=backend['physics_config'],inputs=inputs,sources=sources,policy=POLICY);save(p,spec);specs.append(p)
 save(out/'PROTOCOL.json',dict(schema='anydex_boya_s40_settling_v1',policy=POLICY,groups=list(map(str,specs)),workers=args.workers,input_sha256=inputs,source_sha256=sources,all_objects_train=True,EGAD_objects=0,grasp_attempts=0))
 env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',CANONDEX_COLLISION_PARTITION_CACHE=str(ROOT/'outputs/boya_collision_partition_cache'))
 def one(p):
  with p.with_suffix('.log').open('w')as log:proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker-spec',str(p)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  return dict(spec=str(p),returncode=proc.returncode,output=read(p)['output'])
 with ThreadPoolExecutor(max_workers=args.workers)as pool:receipts=list(pool.map(one,specs))
 rows=[];errors=[];ticks=0;objsteps=0
 for receipt in receipts:
  path=Path(receipt['output'])/'RESULTS.json'
  if receipt['returncode']or not path.exists():errors.append(receipt);continue
  x=read(path);ticks+=x['physics_ticks'];objsteps+=x['object_steps'];rows.extend(x['objects'])
 result=dict(schema='anydex_boya_s40_settling_results_v1',status='complete'if len(rows)==40 and not errors else'infrastructure_error',complete=len(rows)==40 and not errors,ready_objects=sum(r['ready']for r in rows),objects=rows,errors=errors,physics_ticks=ticks,object_steps=objsteps,grasp_attempts=0,policy=POLICY,input_sha256=inputs,source_sha256=sources)
 save(out/'RESULTS.json',result);verify(inputs);verify(sources)
 if len(rows)!=40 or errors:raise RuntimeError('Settling worker failed; evidence retained, no observations or trials dispatched')
 if any(not r['ready']for r in rows):raise RuntimeError('Finite stable-placement criterion not met; no source poses silently accepted')
 byid={r['object_id']:r for r in rows};manifest=copy.deepcopy(scene)
 import trimesh
 for row in manifest['objects']:
  s=byid[row['object_id']];T=np.asarray(s['world_T_object']);source_pose=row['pose_path'];newpose=out/'poses'/(row['object_id']+'.json');q=np.asarray(s['measured_settled_pose_xyzw'][3:]);save(newpose,[[*T[:3,3],q[3],*q[:3]]]);row.update(source_pre_settle_pose_path=source_pose,pose_path=str(newpose),pose_sha256=sha(newpose));mesh=trimesh.load_mesh(row['mesh_path'],process=False);mesh.apply_transform(T)
  row.update(world_T_object=T.tolist(),world_pose_xyzw=s['measured_settled_pose_xyzw'],world_AABB_m=[mesh.vertices.min(0).tolist(),mesh.vertices.max(0).tolist()],world_extents_m=np.ptp(mesh.vertices,axis=0).tolist(),settling_ready=True,settling=s,source_placement_rule='Measured dynamically settled pose; independent trial cold reset with declared zero velocity, not continuous live state. Original native-axis placement retained in settling source manifest.')
 manifest['settling_provenance']=dict(complete=True,ready_objects=40,results_json=str(out/'RESULTS.json'),results_sha256=sha(out/'RESULTS.json'),input_sha256=inputs,source_sha256=sources,entry_boundary='independent cold reset at measured settled pose, zero velocity; not continuous scene')
 manifest.update(physics_steps=objsteps,initialization_object_steps=objsteps,initialization_physics_ticks=ticks,grasp_attempts=0);manifest['source_scene_manifest']=str(args.scene_manifest.resolve());manifest['source_scene_manifest_sha256']=sha(args.scene_manifest);save(out/'objects.json',manifest)
 print(json.dumps(dict(status='complete',ready_objects=40,object_steps=objsteps,grasp_attempts=0,manifest=str(out/'objects.json')),indent=2),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-manifest',type=Path);p.add_argument('--previous-collection',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--workers',type=int,default=4,choices=range(1,5));p.add_argument('--worker-spec',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
 if a.worker_spec:return worker(a.worker_spec)
 if not all([a.scene_manifest,a.previous_collection,a.output_dir]):p.error('source scene, previous collection and fresh output required')
 a.scene_manifest=a.scene_manifest.resolve();a.previous_collection=a.previous_collection.resolve();run(a)
if __name__=='__main__':main()
