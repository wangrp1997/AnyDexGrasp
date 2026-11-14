"""Exact source triangle diagnostics for fixed failed paths; no admission change."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import sys,json,hashlib,time
import numpy as np
import trimesh,coal
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');OUT=ROOT/'outputs/boya_author_pipeline_fourtype_20261114_v2'
sys.path[:0]=['/home/wangrenpeng/CanonDex','/home/wangrenpeng/CanonDex/scripts/research']
from canondex.planning.model import PlanningModel
from canondex.grasp.contact_relation.collision_surface import coal_convex
from run_template_arm_positive import object_asset
start=time.monotonic()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bvh(mesh):
 value=coal.BVHModelOBBRSS();value.beginModel(len(mesh.faces),len(mesh.vertices))
 vertices=coal.StdVec_Vec3s();faces=coal.StdVec_Triangle()
 for p in np.asarray(mesh.vertices):vertices.append(p)
 for face in np.asarray(mesh.faces):faces.append(coal.Triangle(int(face[0]),int(face[1]),int(face[2])))
 value.addSubModel(vertices,faces);value.endModel();return value
I=coal.Transform3s()
def T(frame):return coal.Transform3s(np.asarray(frame)[:3,:3],np.asarray(frame)[:3,3])
req=coal.CollisionRequest();req.num_max_contacts=1
req.enable_contact=False
def intersects(a,ta,b,tb):
 result=coal.CollisionResult();coal.collide(a,ta,b,tb,req,result);return bool(result.isCollision())
dreq=coal.DistanceRequest(True,0.,1e-10);dreq.enable_signed_distance=False;dreq.gjk_tolerance=1e-10
def surface_distance(a,ta,b,tb):
 result=coal.DistanceResult();coal.distance(a,ta,b,tb,dreq,result)
 return max(0.,float(result.min_distance))
sreq=coal.DistanceRequest(True,0.,1e-10);sreq.enable_signed_distance=True;sreq.gjk_tolerance=sreq.epa_tolerance=1e-10
def signed_proxy_distance(a,ta,b,tb):
 result=coal.DistanceResult();coal.distance(a,ta,b,tb,sreq,result);return float(result.min_distance)
queue=read(OUT/'QUEUE_PROTOCOL.json');scene=read(queue['inputs']['scene']['path']);audit=read(OUT/'INDEPENDENT_PREPARATION_AUDIT.json')
rawpoint=np.asarray(audit['independent_domain_and_source_rows']['raw_world_selected_translation_m'])
nearest=[];mesh_locks={}
for row in scene['objects']:
 mesh=trimesh.load_mesh(row['source_mesh'],process=False);mesh.apply_scale(row['scale']);W=np.asarray(row['world_T_object']);local=(rawpoint-W[:3,3])@W[:3,:3]
 p,dist,triangle=trimesh.proximity.closest_point(mesh,local.reshape(1,3));point=p[0]@W[:3,:3].T+W[:3,3]
 nearest.append(dict(object_id=row['object_id'],distance_to_actual_source_surface_m=float(dist[0]),closest_source_point_world_m=point.tolist(),source_triangle_index=int(triangle[0]),source_mesh_sha256=sha(row['source_mesh'])))
 mesh_locks[row['source_mesh']]=sha(row['source_mesh'])
primary=next(x for x in scene['objects'] if x['object_id']==scene['target_object_id']);W=np.asarray(primary['world_T_object']);obj=object_asset(SimpleNamespace(object_root=Path(primary['object_root'])),W,primary['scale']);visual=obj.mesh;visual_bvh=bvh(visual);proxy_shapes=[coal_convex(p) for p in obj.collision_parts]
model=None;hulls={};hand_triangles={};results=[]
for g in [1,6,7,8]:
 d=OUT/f'prepare_g{g}/arm/native';plan=read(d/'plan.json');geometry=read(d/'path_geometry.json');v=geometry['violations']
 if model is None:
  model=PlanningModel(Path(plan['arguments']['robot_urdf']))
 with np.load(d/'commands.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 names=list(map(str,arrays['joint_names']));ph=arrays['phase'];selected=[]
 for phase in ['home_to_pregrasp','approach']:
  rows=[x for x in v if ph[x['sample']]==phase]
  if not rows:continue
  for tag,key in [('first',lambda x:x['sample']),('deepest',lambda x:x['distance_m']),('last',lambda x:-x['sample'])]:
   row=min(rows,key=key);selected.append((phase,tag,row))
 records=[]
 for phase,tag,row in selected:
  sample=row['sample'];link=row['link'];q=model.full_state(dict(zip(names,map(float,arrays['joint_target'][sample]))));frame=model.fk.transforms(q)[link]
  if link not in hulls:
   collision=model.collision_mesh(link);hulls[link]=coal_convex(collision.convex_hull)
  if link not in hand_triangles:hand_triangles[link]=bvh(model.collision_mesh(link))
  hull_intersect=intersects(visual_bvh,T(W),hulls[link],T(frame))
  exact_intersect=intersects(visual_bvh,T(W),hand_triangles[link],T(frame))
  records.append(dict(phase=phase,representative=tag,sample=sample,time_s=float(arrays['time_s'][sample]),link=link,proxy_object_part=row['other'],proxy_signed_distance_m=row['distance_m'],actual_object_triangle_surface_vs_hand_collision_hull_intersects=hull_intersect,actual_object_triangle_surface_vs_hand_source_collision_triangles_intersects=exact_intersect,actual_object_surface_to_hand_hull_unsigned_distance_m=0. if hull_intersect else surface_distance(visual_bvh,T(W),hulls[link],T(frame)),hand_link_origin_world_m=frame[:3,3].tolist()))
 endpoint=[]
 for sample in [799,1299]:
  q=model.full_state(dict(zip(names,map(float,arrays['joint_target'][sample]))));frames=model.fk.transforms(q);distances=[]
  for link in model.collision_links:
   if link not in hulls:hulls[link]=coal_convex(model.collision_mesh(link).convex_hull)
   for part,shape in enumerate(proxy_shapes):distances.append((signed_proxy_distance(hulls[link],T(frames[link]),shape,T(W)),link,part))
  distance,link,part=min(distances)
  endpoint.append(dict(sample=sample,phase=str(ph[sample]),time_s=float(arrays['time_s'][sample]),minimum_all_robot_hulls_vs_primary_proxy_signed_distance_m=distance,nearest_link=link,nearest_proxy_part=part,actual_source_object_surface_to_nearest_hand_hull_unsigned_distance_m=surface_distance(visual_bvh,T(W),hulls[link],T(frames[link]))))
 results.append(dict(gesture_id=g,representatives=records,endpoints=endpoint,deepest_original_triangle_intersections_in_violating_phases=all(x['actual_object_triangle_surface_vs_hand_source_collision_triangles_intersects'] for x in records if x['representative']=='deepest')))
for p,d in mesh_locks.items():assert sha(p)==d
output=dict(schema='anydex_boya_author_fixed_path_triangle_geometry_audit_v1',status='passed',scope='Post hoc diagnostic only: actual source triangle meshes versus unchanged hand collision source triangles/hulls at fixed path representative states; source ground truth never alters candidate, plan or label.',primary_object_id=primary['object_id'],primary_visual_source_triangles=len(visual.faces),primary_visual_source_watertight=bool(visual.is_watertight),primary_collision_proxy_parts=len(proxy_shapes),raw879_point_world_m=rawpoint.tolist(),raw879_actual_source_surface_distances=nearest,results=results,source_mesh_bindings=mesh_locks,physics_steps=0,new_planning_calls=0,admission_threshold_changes=0,wall_s=time.monotonic()-start,limitations='Representative states do not certify every proxy violation against exact triangles or continuous clearance. Positive actual-source surface intersections at deepest states establish that the complete failed segments cannot be explained solely by proxy exterior bulges. No force or physical grasp conclusion.')
(OUT/'INDEPENDENT_GEOMETRY_AUDIT.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({'status':output['status'],'nearest':nearest,'results':results,'wall_s':output['wall_s']},indent=2))
