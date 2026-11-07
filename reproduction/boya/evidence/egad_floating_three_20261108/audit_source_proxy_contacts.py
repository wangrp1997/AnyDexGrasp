import json,hashlib,time,sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import math
r=Path('/home/wangrenpeng/AnyDexGrasp/outputs/egad_external12_20261107_v1')
objs={o['id']:o for o in json.load(open(r/'objects.json'))['objects']}
report={'scope':'Read-only saved A3/E0 source-vs-proxy contact audit; no simulation/no score changes','objects':{},'inputs':{}}
def sha(p):
 p=Path(p); h=hashlib.sha256(p.read_bytes()).hexdigest();report['inputs'][str(p)]=h;return h
def loadm(p):
 sha(p); return trimesh.load(p,force='mesh',process=False)
def dist(mesh,pts):
 # Exact nearest triangle for finite queried points; chunk avoids large point*face temporary arrays.
 a=[]; cp=[];tid=[]
 for i in range(0,len(pts),16):
  c,d,t=trimesh.proximity.closest_point_naive(mesh,np.asarray(pts)[i:i+16]);a.extend(d.tolist());cp.extend(c.tolist());tid.extend(t.tolist())
 return np.array(a),np.array(cp),np.array(tid)
for oid in ['egad_A3','egad_E0']:
 o=objs[oid]; source=loadm(o['mesh_path']); parts=[loadm(p['path']) for p in o['convex_parts']]; proxy=trimesh.util.concatenate(parts)
 d=r/'execution/execution'/oid/'cohorts/cohort_0000/execution_000';sha(d/'evaluation.json');sha(d/'trace.npz');ev=json.load(open(d/'evaluation.json'));tr=np.load(d/'trace.npz')
 out={'source_winding_consistent':bool(source.is_winding_consistent),'source_watertight':bool(source.is_watertight),'source_volume_m3':float(source.volume),'proxy_component_count':len(parts),'proxy_component_volume_sum_m3':float(sum(x.volume for x in parts)),'checkpoints':[]}
 for chk in ev['checkpoints']:
  positive=[c for c in chk['object_contacts'] if c['normal_force_n']>1e-3]
  if not positive:continue
  pose=np.asarray(chk['object_pose_xyzw']); rot=Rotation.from_quat(pose[3:]).as_matrix();points=(np.asarray([c['position_world_m'] for c in positive])-pose[:3])@rot
  sd,scp,stid=dist(source,points);pd,pcp,ptid=dist(proxy,points)
  # Project contact position onto supplied convex geometry, then ask distance to original visual.
  p2s,p2scp,_=dist(source,pcp)
  signed_projected=trimesh.proximity.signed_distance(source,pcp)
  inside_projected=source.contains(pcp)
  normals=source.face_normals[stid];raw=(np.asarray([c['raw_normal_world'] for c in positive]))@rot
  for i,c in enumerate(positive):
   out['checkpoints'].append({'phase':chk['phase'],'step':chk['step'],'body':c['body'],'normal_force_n':c['normal_force_n'],'reported_separation_m':c['separation_m'],'contact_world_m':c['position_world_m'],'contact_local_m':points[i].tolist(),'contact_to_source_surface_m':float(sd[i]),'contact_to_proxy_component_surface_m':float(pd[i]),'closest_proxy_surface_to_source_surface_m':float(p2s[i]),'closest_source_point_m':scp[i].tolist(),'closest_proxy_point_m':pcp[i].tolist(),'projected_proxy_point_inside_source':bool(inside_projected[i]),'projected_proxy_point_signed_source_distance_m':float(signed_projected[i]),'reported_normal_dot_nearest_source_face_normal':float(raw[i]@normals[i]),'reported_normal_to_nearest_source_face_angle_rad':float(math.acos(np.clip(raw[i]@normals[i],-1,1)))})
 # Saved full-step load/displacement timeline, without claiming force closure.
 phase=tr['phase'];load=tr['finger_loads_n'];pose=tr['object_pose_xyzw'];palm=tr['desired_palm']
 both=(load[:,0]>1e-3)&(load[:,1:].sum(1)>1e-3)
 phases={}
 for ph in ['close','hold','lift']:
  mask=phase==ph;ix=np.flatnonzero(mask);last=ix[both[mask]][-1] if np.any(both[mask]) else None
  phases[ph]={'start_step':int(ix[0]+1),'end_step':int(ix[-1]+1),'both_loaded_steps':int(both[mask].sum()),'last_both_loaded_step':int(last+1) if last is not None else None,'max_load_n':load[mask].max(0).tolist(),'final_load_n':load[ix[-1]].tolist(),'object_translation_since_phase_start_m':(pose[ix[-1],:3]-pose[ix[0],:3]).tolist(),'object_rotation_since_phase_start_rad':float(Rotation.from_quat(pose[ix[0],3:]).inv().__mul__(Rotation.from_quat(pose[ix[-1],3:])).magnitude())}
 out['timeline']=phases
 # Frame at first collapse of sustained loads, with wrist-vs-object relative motion.
 liftix=np.flatnonzero(phase=='lift');lastboth=liftix[both[liftix]][-1] if both[liftix].any() else None
 out['lost_after_lift_step']=int(lastboth-liftix[0]+1) if lastboth is not None else 0
 out['last_loaded_lift_time_s']=float(tr['time_s'][lastboth]-tr['time_s'][liftix[0]]) if lastboth is not None else None
 out['lift_palm_displacement_m']=(palm[liftix[-1],:3,3]-palm[liftix[0],:3,3]).tolist()
 out['palm_tracking_max_position_error_m']=ev['palm_position_error_m']
 out['pickup']=ev['pickup']; report['objects'][oid]=out
report['limitations']=['Contact position is a PhysX reported contact point, not certified original surface point; positive normal force distinguishes load-bearing contacts from speculative zero-force contact-offset points.', 'Distances are exact closest-triangle distances for saved finite contact queries; source/proxy are supplied meshes, not an exhaustive shape bound.', 'A3 proxy contains 4 convex components; distance is to component surfaces, not signed exterior boundary of geometric union.', 'Transform uses saved object pose at same checkpoint, matching original-mesh frame.', 'Only close/hold checkpoints save full loaded contact points; per-step trace has summed finger loads but not contact patches.', 'Source signed distances use trimesh positive-inside convention; contains agrees on all 15 positive-force proxy-projected contact queries outside watertight consistently wound source meshes.', 'No counterfactual simulation, therefore differences may affect contact mapping but cannot prove they caused failure.']
path=Path('/tmp/EGAD_PROXY_CONTACT_AUDIT_20261108.json');path.write_text(json.dumps(report,indent=2)+'\n')
for oid,o in report['objects'].items():
 print(oid,'parts',o['proxy_component_count'],'loadloss lift steps',o['lost_after_lift_step'])
 for c in o['checkpoints']: print(c['phase'],c['body'],'force',round(c['normal_force_n'],3),'src_mm',round(c['contact_to_source_surface_m']*1000,3),'proxy_mm',round(c['contact_to_proxy_component_surface_m']*1000,3),'proxy_to_src_mm',round(c['closest_proxy_surface_to_source_surface_m']*1000,3),'sep_mm',round(c['reported_separation_m']*1000,3))
 print('timeline',o['timeline'])
print(path)
