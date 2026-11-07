"""RTX replay of actual hand-only traces, original source visual triangle meshes.

Only saved native body/object states are rendered. No dynamics, no posed guess,
no replacement successful grasp. Both failures and successes are displayed.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');C=Path('/home/wangrenpeng/CanonDex')
sys.path[:0]=[str(C),str(C/'scripts/research')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def run(a):
 import numpy as np, torch, trimesh, xml.etree.ElementTree as ET
 from scipy.spatial.transform import Rotation
 from PIL import Image,ImageDraw,ImageFont
 from pxr import UsdGeom,Usd,Gf
 import isaaclab.sim as sim_utils
 from canondex.scene.floor import spawn_floor_and_light
 from canondex.scene.camera import make_cameras,fitted_view,rgb_frame
 from canondex.robots.kinematics import pose_matrix,UrdfKinematics
 from canondex.scene.render_sync import matrix_errors
 began=time.monotonic();protocol=read(a.run/'protocol.json');results=read(a.run/'RESULTS.json')
 if results['status']!='complete' or sha(a.run/'protocol.json')!=results['protocol_sha256']:raise ValueError('Completed bound three-case run required')
 out=a.run/'videos'
 if out.exists():raise ValueError('Do not overwrite media')
 out.mkdir();sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.004,device='cpu',gravity=(0.,0.,0.),use_fabric=False,visualizer_cfgs=[]))
 spawn_floor_and_light();cameras=make_cameras();sim.reset();steps0=sim.get_physics_step_count();all_reports=[]
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 def font(size):return ImageFont.truetype(fontpath,size)
 def author(path,mesh,color):
  g=UsdGeom.Mesh.Define(sim.stage,path);g.CreatePointsAttr(np.asarray(mesh.vertices).tolist());g.CreateFaceVertexCountsAttr([3]*len(mesh.faces));g.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces).ravel().tolist());g.CreateSubdivisionSchemeAttr('none');g.CreateDisplayColorAttr([Gf.Vec3f(*color)]);g.CreateDoubleSidedAttr(True)
 def xform(path):
  v=UsdGeom.Xform.Define(sim.stage,path);return v.AddTransformOp()
 def setmatrix(op,T):op.Set(Gf.Matrix4d(np.asarray(T,dtype=float).T.tolist()))
 def checkmatrix(path,T):
  v=np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(sim.stage.GetPrimAtPath(path)),dtype=float).T
  return matrix_errors(v,T)
 for row in results['results']:
  oid=row['object_id'];execution=Path(row['evaluation']).parent;e=read(row['evaluation']);job=read(a.run/'jobs'/(oid+'.json'))
  if sha(row['evaluation'])!=row['evaluation_sha256']or sha(execution/'trace.npz')!=e['trace_sha256']:raise ValueError('Trace/evaluation binding changed')
  with np.load(execution/'trace.npz',allow_pickle=False)as z:t={k:z[k].copy()for k in z.files}
  sim.stage.RemovePrim('/World/Playback');UsdGeom.Xform.Define(sim.stage,'/World/Playback')
  target=out/oid;target.mkdir();ops={};mesh_sources={}
  source=Path(job['hand_urdf']);tree=ET.parse(source).getroot();colors={}
  for material in tree.findall('material'):
   color=material.find('color')
   if color is not None:colors[material.get('name')]=[float(v)for v in color.get('rgba').split()[:3]]
  bn=list(t['body_names']);bodyops={};bodyids={};fk=UrdfKinematics(execution/'boya_translation_gantry.urdf')
  # The hand subtree has original palm/finger rigid bodies. Empty gantry bodies
  # have no visual and are intentionally absent from the image.
  for link in tree.findall('link'):
   name=link.get('name');visuals=link.findall('visual')
   if not visuals or name not in fk.links:continue
   owner,owner_T_link=fk.fixed_attachment(name,np.eye(4))
   if owner not in bn:raise ValueError('Visual body owner missing: '+owner)
   base='/World/Playback/'+owner
   if owner not in bodyops:bodyops[owner]=xform(base);bodyids[owner]=bn.index(owner)
   for j,v in enumerate(visuals):
    meshnode=v.find('geometry/mesh')
    if meshnode is None:continue
    mp=(source.parent/meshnode.get('filename')).resolve();mesh=trimesh.load_mesh(mp,process=False)
    mesh.apply_scale([float(x)for x in meshnode.get('scale','1 1 1').split()])
    o=v.find('origin');T=np.eye(4)
    if o is not None:
     T[:3,3]=[float(x)for x in o.get('xyz','0 0 0').split()];T[:3,:3]=Rotation.from_euler('xyz',[float(x)for x in o.get('rpy','0 0 0').split()]).as_matrix()
    mesh.apply_transform(owner_T_link@T);mat=v.find('material');color=colors.get(mat.get('name'),[.55,.58,.62])if mat is not None else[.55,.58,.62]
    author(base+f'/visual_{name}_{j}',mesh,color);mesh_sources[str(mp)]=sha(mp)
  # Every collidable original hand link must remain visually represented.
  expected={fk.fixed_attachment(l.get('name'),np.eye(4))[0]for l in tree.findall('link')if l.find('collision')is not None and l.get('name')in fk.links}
  if not expected<=set(bodyops):raise ValueError('Missing visible collidable body: '+str(expected-set(bodyops)))
  objmesh=trimesh.load_mesh(job['object_row']['mesh_path'],process=False);objectop=xform('/World/Playback/object');author('/World/Playback/object/visual',objmesh,[.90,.34,.10])
  mesh_sources[job['object_row']['mesh_path']]=sha(job['object_row']['mesh_path'])
  if 'body_pose_xyzw'not in t:t['body_pose_xyzw']=t['body_pose']
  points=t['body_pose_xyzw'][:,[bodyids[n]for n in bodyops],:3].reshape(-1,3)
  points=np.vstack((points,t['object_pose_xyzw'][:,:3]))
  setups={}
  for name,direction in [('overview',(1.,-1.,.7)),('hand',(-1.,.2,.9))]:
   eye,look=fitted_view(points,padding=.055,direction=direction,minimum_distance=.25)
   cameras[name].set_world_poses_from_view(torch.tensor(eye[None],dtype=torch.float32,device=cameras[name].device),torch.tensor(look[None],dtype=torch.float32,device=cameras[name].device));setups[name]=dict(eye=eye.tolist(),target=look.tolist())
  # 2100 x .004s =8.4s ->210 video frames at25fps plus1s final hold.
  indices=np.minimum(np.rint(np.arange(1,211)*.04/.004).astype(int)-1,len(t['time_s'])-1)
  if len(t['time_s'])!=2100:indices=np.unique(np.minimum(np.rint(np.arange(.04,float(t['time_s'][-1])+.0001,.04)/.004).astype(int)-1,len(t['time_s'])-1))
  frames=[];maximum=np.zeros(3);first={};motion={n:0. for n in cameras}
  for fi,index in enumerate(indices):
   for name,op in bodyops.items():
    T=pose_matrix(t['body_pose_xyzw'][index,bodyids[name]]);setmatrix(op,T);maximum=np.maximum(maximum,checkmatrix('/World/Playback/'+name,T))
   T=pose_matrix(t['object_pose_xyzw'][index]);setmatrix(objectop,T);maximum=np.maximum(maximum,checkmatrix('/World/Playback/object',T))
   for _ in range(2 if fi else 12):
    sim.render()
    for camera in cameras.values():camera.update(.004)
   canvas=Image.new('RGB',(1280,560),'#edf1f5');draw=ImageDraw.Draw(canvas)
   outcome='LIFTED'if e['task_success']else'FAILED'
   draw.text((14,10),f'{oid}  |  Hand-only PhysX diagnostic  |  {outcome}  |  {t["phase"][index]}  {t["time_s"][index]:.2f}s',font=font(22),fill='#183047')
   for col,(name,camera)in enumerate(cameras.items()):
    arr=rgb_frame(camera);canvas.paste(Image.fromarray(arr),(640*col,48))
    if name not in first:first[name]=arr.copy()
    motion[name]=max(motion[name],float(abs(arr.astype(float)-first[name]).mean()))
   holds=np.flatnonzero(t['phase']=='hold');reference=int(holds[-1])if len(holds)else 0
   prelift=t['object_pose_xyzw'][reference,2]
   rise=(float(t['object_pose_xyzw'][index,2])-float(prelift))*1000
   draw.text((14,531),f'Original source visual meshes | Native recorded states | Object rise {rise:.1f} mm | 25 fps, real time',font=font(17),fill='#183047')
   frames.append(canvas)
   if fi in [0,len(indices)//2,len(indices)-1]:canvas.save(target/f'frame_{index:04d}.png')
   if fi%50==0:print('FLOAT_RTX',oid,fi,len(indices),flush=True)
  if max(maximum)>1e-6 or min(motion.values())<.1:raise ValueError('Replay transform or motion verification failed')
  gif=target/'hand_only.gif';mp4=target/'hand_only.mp4'
  durations=[40]*len(frames);durations[-1]+=1000
  frames[0].save(gif,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
  encoder=C/'docs/dev/grasp/evidence/20261025/two_sided_template_video/encode_saved_gif.py'
  with (target/'encoding.log').open('w')as log:subprocess.run([sys.executable,'-B',str(encoder),str(gif),str(mp4)],stdout=log,stderr=subprocess.STDOUT,check=True)
  report=dict(complete=True,object_id=oid,task_success=bool(e['task_success']),renderer='Isaac Sim RTX source-visual playback',
   source_trace_sha256=e['trace_sha256'],source_evaluation_sha256=row['evaluation_sha256'],
   physics_steps_during_replay=sim.get_physics_step_count()-steps0,visual_mesh_sha256=mesh_sources,
   source_visual_meshes=True,root_and_fingers_from_actual_recorded_native_body_states=True,
   max_render_matrix_errors=maximum.tolist(),camera_setups=setups,pixel_motion=motion,frames=len(indices),fps=25,final_pause_s=1.,video_sha256=sha(mp4))
  if report['physics_steps_during_replay']:raise ValueError('Unexpected replay integration')
  save(target/'render_check.json',report);all_reports.append(report)
 title='Boya hand-only diagnosis: A6, A3, E0';parts=['<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:18px system-ui;max-width:1300px;margin:30px auto;background:#eef1f5}video{width:100%}.case{background:white;padding:20px;margin:20px 0}</style><h1>'+title+'</h1><p>Frozen candidates; one physical trial per case. Diagnostic only; original external result remains 1/12.</p>']
 import html
 for report in all_reports:
  oid=report['object_id'];label='成功'if report['task_success']else'未抓起'
  parts.append(f'<section class="case"><h2>{oid} — {label}</h2><video controls preload="metadata" src="{oid}/hand_only.mp4"></video>')
  case=next(c for c in protocol['cases']if c['object_id']==oid)
  if case.get('old_execution'):
   old=ROOT/'outputs/egad_external12_20261107_v1/execution/videos'/oid/'isaac_full_arm_pickup.mp4';relative=os.path.relpath(old,out)
   parts.append('<p>原完整机械臂轨迹：</p><video controls preload="metadata" src="'+html.escape(relative)+'"></video>')
  parts.append('</section>')
 (out/'index.html').write_text('\n'.join(parts));save(out/'RESULTS.json',dict(complete=True,cases=all_reports,wall_s=time.monotonic()-began,physics_steps_during_replay=sim.get_physics_step_count()-steps0));print('FLOAT_VIDEOS_COMPLETE',flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve()
 os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
 from isaaclab.app import AppLauncher
 launcher=AppLauncher(headless=True,device='cpu',enable_cameras=True,visualizer=None,kit_args='--/log/file=/dev/null --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false')
 code=1
 try:run(a);code=0
 finally:launcher.app.close(exit_code=code)
if __name__=='__main__':main()
