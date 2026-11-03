"""Read-only attribution of saved original DH3 collision rejections.

Segmentation is used only after inference for diagnosis. Saved RGB-D pixels are
bit-bound to the original quantized scene points. Optional selected-pose CPU
geometry replays diagnose original voxel/PCL lineage; no network or physics.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io


def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rec(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def require(v,msg):
 if not v:raise AssertionError(msg)
def summary(x):
 x=np.asarray(x);return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if len(x) else None

def classify(scene,scene_dir,run_dir):
 points=np.load(run_dir/'frame/scene_points_down.npy',allow_pickle=False)
 depth=np.asarray(Image.open(scene_dir/'depth.png'));meta=scipy.io.loadmat(scene_dir/'meta.mat')
 K=np.asarray(meta['intrinsic_matrix'],dtype=np.float64);factor=float(np.asarray(meta['factor_depth']).reshape(-1)[0])
 require(np.array_equal(K,scene['camera']['K']),'Camera K mismatch')
 require(factor==scene['spec']['camera']['depth_factor'],'Depth factor mismatch')
 seg=np.load(scene_dir/'segmentation_for_validation_only.npy',allow_pickle=False)
 uv=np.rint(np.column_stack((K[0,0]*points[:,0]/points[:,2]+K[0,2],K[1,1]*points[:,1]/points[:,2]+K[1,2]))).astype(np.int64)
 require(((uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])).all(),'Invalid source pixels')
 u,v=uv.T;z=depth[v,u]/factor
 rebuilt=np.column_stack(((u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z)).astype(np.float32)
 require(rebuilt.dtype==points.dtype and rebuilt.tobytes()==points.tobytes(),'Saved points are not exact original depth pixels')
 labels=seg[v,u]&((1<<24)-1)
 T=np.asarray(scene['camera']['T_WC'],dtype=np.float64)
 require(np.allclose(T@scene['camera']['T_CW'],np.eye(4),atol=1e-12,rtol=0),'Camera inverse')
 world=points.astype(np.float64)@T[:3,:3].T+T[:3,3]
 ids=scene['body_ids'];table=labels==ids['table'];obj=labels==ids['object']
 require((table|obj).all(),'Unexpected visible scene category')
 topz=scene['spec']['table']['position_world_m'][2]+scene['spec']['table']['size_m'][2]/2
 require(abs(world[table,2]-topz).max()<.00051,'Table surface/units mismatch')
 with np.load(run_dir/'collision/collision_outputs.npz',allow_pickle=False) as s:c={k:s[k].copy() for k in s.files}
 masks=np.unpackbits(c['scene_inclusion_masks_packed'],axis=1)[:,:len(points)].astype(bool)
 require(np.array_equal(masks.sum(1),c['overlap_point_counts']),'Saved packed mask/count mismatch')
 require(np.array_equal(c['admission_mask'],masks.sum(1)==0),'Saved original admission mismatch')
 counts=np.column_stack((masks[:,table].sum(1),masks[:,obj].sum(1)))
 return dict(points=points,world=world,labels=labels,table=table,obj=obj,uv=uv,c=c,masks=masks,counts=counts,K=K,T=T,
  binding=dict(source_pixels_bit_exact=True,point_units='metres',collision_coordinates='camera',diagnostic_coordinates='world',
   T_WC=T.tolist(),depth_factor=factor,table_top_world_z_m=topz,table_saved_world_z_m=summary(world[table,2]),
   original_scene_point_count=len(points),segmented_table_points=int(table.sum()),segmented_object_points=int(obj.sum()),
   segmentation_use='Saved renderer ID segmentation for post-inference diagnosis only; not provided to the network or gate',
   inputs=[rec(scene_dir/f) for f in ('scene_manifest.json','depth.png','meta.mat','segmentation_for_validation_only.npy')]+
          [rec(run_dir/'frame/scene_points_down.npy'),rec(run_dir/'collision/collision_outputs.npz')]))

def candidate(i,data):
 h=data['c']['DH3_grasp_group_array'][i];two=data['c']['two_finger_grasp_group_array'][i];R=two[4:13].reshape(3,3).astype(np.float64)
 direction=R[:,0];direction/=np.linalg.norm(direction);worlddir=data['T'][:3,:3]@direction
 base=data['T'][:3,:3]@(h[12:15]+direction*h[1])+data['T'][:3,3]
 seed=data['T'][:3,:3]@two[13:16]+data['T'][:3,3]
 return dict(index=int(i),pre_collision_index=int(data['c']['source_indices'][i]),raw_learned_score=float(h[0]),hand_type=int(h[2]),
  mapped_hand_width_m=float(h[18]),expanded_two_finger_width_m=float(two[1]),two_finger_base_depth_m=float(two[3]),
  full_DH3_depth_m=float(h[1]),learned_additional_depth_m=float(h[1]-two[3]),position_register=float(h[15]),rotation_register=float(h[16]),
  approach_unit_camera=direction.tolist(),approach_unit_world=worlddir.tolist(),angle_from_world_down_degrees=float(np.rad2deg(np.arccos(np.clip(-worlddir[2],-1,1)))),
  original_two_finger_seed_world_m=seed.tolist(),mapped_hand_origin_after_full_depth_world_m=base.tolist(),
  overlaps_total=int(data['counts'][i].sum()),overlap_table_points=int(data['counts'][i,0]),overlap_object_points=int(data['counts'][i,1]))

class Geometry:
 def __init__(self,repo):
  import open3d as o3d
  self.o=o3d;self.repo=repo;self.cache={};self.cpu_grid_queries=0;self.triangle_distance_queries=0;self.occupancy_queries=0
  tree=ast.parse((repo/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py').read_text())
  self.mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='joint_mesh_mapping' for t in n.targets))
  root=repo/'generate_mesh_and_pointcloud/dh3_urdf/dh3_urdf/meshes'
  self.source_components=[]
  for k in ['base']+[str(i) for i in range(12)]:
   p=root/self.mapping[k];mesh=o3d.io.read_triangle_mesh(str(p))
   self.source_components.append(dict(link=self.mapping[k],triangles=len(mesh.triangles),source=rec(p)))
 def slot(self,key,collision):
  if key in self.cache:return self.cache[key]
  o=self.o;asset=Path(collision['assets']['root']);slot=read(asset/'slot_receipts'/(key+'.json'))
  r=next(v for v in collision['assets']['slots'] if v['key']==key);require(sha(r['path'])==r['sha256'],'PCL changed')
  meshrec=slot['output_mesh'];require(sha(meshrec['path'])==meshrec['sha256'],'Generated STL changed')
  cloud=o.io.read_point_cloud(r['path']);mesh=o.io.read_triangle_mesh(meshrec['path'])
  require(sum(r['triangles'] for r in self.source_components)==len(mesh.triangles),'Original component triangle count mismatch')
  allscene=o.t.geometry.RaycastingScene(nthreads=2);allscene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(mesh))
  links=[];start=0;tri=np.asarray(mesh.triangles);verts=np.asarray(mesh.vertices)
  for part in self.source_components:
   count=part['triangles'];piece=o.geometry.TriangleMesh(o.utility.Vector3dVector(verts.copy()),o.utility.Vector3iVector(tri[start:start+count].copy()));piece.remove_unreferenced_vertices();start+=count
   water=bool(piece.is_watertight());rs=None
   if water:
    rs=o.t.geometry.RaycastingScene(nthreads=2);rs.add_triangles(o.t.geometry.TriangleMesh.from_legacy(piece))
   from scipy.spatial import ConvexHull
   # Exact-vertex convex envelope can only certify OUTSIDE material. Contained
   # points are not claimed to collide with the source's non-convex solid.
   hull=ConvexHull(np.asarray(piece.vertices))
   links.append(dict(name=part['link'],watertight=water,scene=rs,hull_planes=hull.equations.copy()))
  result=dict(cloud=cloud,mesh=mesh,allscene=allscene,links=links,source_mesh=meshrec,pcl=r)
  self.cache[key]=result;return result
 def diagnose(self,i,data,collision):
  o=self.o;h=data['c']['DH3_grasp_group_array'][i];two=data['c']['two_finger_grasp_group_array'][i]
  key='pose'+str(int(h[2]))+'_'+str(round(h[18]*100,1));asset=self.slot(key,collision)
  R=h[3:12].reshape(3,3);t=h[12:15];twoR=two[4:13].reshape(3,3);twoT=two[13:16]
  tf=np.vstack((np.hstack((twoR,twoT.reshape(3,1))),[0,0,0,1]))
  direction=(tf@np.array([.01,0,0,1]))[:3]-twoT
  direction=direction/math.sqrt(np.power(direction,2).sum())
  depth_tf=np.eye(4,dtype=np.float32);depth_tf[:3,3]=direction*h[1]
  endpoint=copy.deepcopy(asset['cloud']);endpoint.transform(np.vstack((np.hstack((R,t.reshape(3,1))),[0,0,0,1])));endpoint.transform(depth_tf)
  full=copy.deepcopy(endpoint);shifts=[np.zeros(3)]
  for cm in range(2,int(collision['original_parameters']['approach_distance_m']*100)+1,3):
   shift=np.eye(4,dtype=np.float32);shift[:3,3]=-twoR[:,0]*cm*.01
   shifted=copy.deepcopy(full);shifted.transform(shift);full=full+shifted
   shifts += [a+shift[:3,3].astype(np.float64) for a in shifts]
  require(len(shifts)==4,'Unexpected approach offset recipe')
  size=collision['original_parameters']['hand_and_voxel_grid_m'];down=full.voxel_down_sample(size)
  grid=o.geometry.VoxelGrid.create_from_point_cloud(down,size)
  replay=np.asarray(grid.check_if_included(o.utility.Vector3dVector(data['points'])),dtype=bool);self.cpu_grid_queries+=1
  require(np.array_equal(replay,data['masks'][i]),'Selected-candidate original grid does not reproduce saved mask bits')
  traced,_,indices=full.voxel_down_sample_and_trace(size,full.get_min_bound()-size*.5,full.get_max_bound()+size*.5,False)
  p=np.asarray(down.points);q=np.asarray(traced.points)
  require(p.shape==q.shape,'Trace downsample size differs')
  exact=np.array_equal(p,q)
  if not exact:
   from scipy.spatial import cKDTree
   dist,mapping=cKDTree(q).query(p)
   require(float(dist.max())<1e-12 and len(set(mapping.tolist()))==len(p),'Trace centroids do not correspond')
   indices=[indices[j] for j in mapping]
  n=len(asset['cloud'].points);lineage={}
  for point,idx in zip(p,indices):
   stages=np.unique(np.asarray(idx,dtype=np.int64)//n);bits=sum(1<<int(k) for k in stages)
   cell=tuple(np.floor((point-grid.origin)/size).astype(np.int64));lineage[cell]=lineage.get(cell,0)|bits
  hit=np.flatnonzero(replay);keys=np.floor((data['points'][hit]-grid.origin)/size).astype(np.int64)
  bits=np.array([lineage[tuple(k)] for k in keys],dtype=np.uint8)
  endpoint_contrib=(bits&1)!=0;labels=data['labels'][hit]
  bytype={}
  for name,mask in [('all',np.ones(len(hit),bool)),('table',data['table'][hit]),('object',data['obj'][hit])]:
   bytype[name]=dict(total=int(mask.sum()),endpoint_voxel_contributed=int((mask&endpoint_contrib).sum()),
    approach_only_voxel=int((mask&~endpoint_contrib).sum()),
    contributing_offsets=[dict(nominal_backward_m=off,points=int((mask&((bits&(1<<j))!=0)).sum())) for j,off in enumerate([0,.02,.05,.07])])
  invR=np.linalg.inv(R);hitpoints=data['points'][hit].astype(np.float64);inside=[];distances=[];envelope_violation=[]
  occupancy_available=all(link['watertight'] for link in asset['links'])
  for shift in shifts:
   local=(hitpoints-t-depth_tf[:3,3]-shift)@invR.T
   tensor=o.core.Tensor(local.astype(np.float32),dtype=o.core.Dtype.Float32)
   distances.append(asset['allscene'].compute_distance(tensor).numpy().astype(np.float64));self.triangle_distance_queries+=1
   # Positive max halfspace violation bounds distance from this convex link
   # envelope below. Min over all links still bounds distance from their union.
   per_link=[]
   for link in asset['links']:
    planes=link['hull_planes'];per_link.append(np.max(local@planes[:,:3].T+planes[:,3],axis=1))
   envelope_violation.append(np.min(np.stack(per_link),axis=0))
   if occupancy_available:
    which=np.zeros(len(local),dtype=bool)
    for link in asset['links']:
     which|=link['scene'].compute_occupancy(tensor,nsamples=3).numpy()>.5;self.occupancy_queries+=1
    inside.append(which)
  hull_violation=np.stack(envelope_violation);outside_all_poses=(hull_violation>1e-8).all(0)
  d=np.stack(distances);mesh_result=dict(exact_source_triangle_surface_distances_m=summary(d.min(0)),
   endpoint_surface_distances_m=summary(d[0]),
   source_component_closedness=[dict(name=l['name'],watertight=l['watertight']) for l in asset['links']],
   union_of_per_link_closed_mesh_occupancy_available=occupancy_available,
   boundary='Unsigned distance is to original triangular surfaces, not an AABB. Inside is queried only when every separately assembled source link is watertight; union across links avoids cancellation from intersecting components. This is discrete-pose geometry on depth-quantized observed points, not a physics/contact outcome.')
  mesh_result['convex_link_envelope_certificate']={}
  for name,mask in [('all',np.ones(len(hit),bool)),('table',data['table'][hit]),('object',data['obj'][hit])]:
   mesh_result['convex_link_envelope_certificate'][name]=dict(total=int(mask.sum()),
    definitely_outside_every_link_at_endpoint=int((mask&(hull_violation[0]>1e-8)).sum()),
    definitely_outside_every_link_at_all_four_discrete_poses=int((mask&outside_all_poses).sum()),
    outside_all_poses_halfspace_lower_bound_over_half_mm=int((mask&(hull_violation.min(0)>.0005)).sum()),
    certified_outside_union_distance_lower_bound_m=summary(hull_violation.min(0)[mask&outside_all_poses]))
  mesh_result['convex_envelope_boundary']='Each hull encloses all triangle vertices of one separate original assembled link. Outside every such hull is a sufficient geometric certificate of being outside every link, without assuming watertightness. Inside a hull is inconclusive and is never equated to material collision. Certificates concern these four discrete poses, not the full continuous sweep.'
  continuous={}
  for margin in [0.,.0005]:
   possible_any=np.zeros(len(hitpoints),dtype=bool)
   for before,after in zip(shifts[:-1],shifts[1:]):
    local_start=(hitpoints-t-depth_tf[:3,3]-before)@invR.T
    delta=(after-before)@invR.T
    for link in asset['links']:
     planes=link['hull_planes'];coeff=planes[:,:3]@delta
     c=local_start@planes[:,:3].T+planes[:,3]-margin
     pos=coeff>1e-12;neg=coeff< -1e-12;zero=~(pos|neg)
     low=np.maximum(0.,np.max(c[:,pos]/coeff[pos],axis=1)) if pos.any() else np.zeros(len(c))
     high=np.minimum(1.,np.min(c[:,neg]/coeff[neg],axis=1)) if neg.any() else np.ones(len(c))
     possible=(low<=high)&(~(c[:,zero]>0).any(axis=1) if zero.any() else True)
     possible_any|=possible
   continuous[str(margin)]={name:dict(total=int(mask.sum()),
      certified_outside_all_links_whole_piecewise_linear_approach=int((mask&~possible_any).sum()),
      inconclusive_inside_some_swept_convex_envelope=int((mask&possible_any).sum()))
      for name,mask in [('all',np.ones(len(hit),bool)),('table',data['table'][hit]),('object',data['obj'][hit])]}
  mesh_result['continuous_sweep_convex_envelope_certificate']=dict(
   method='Analytically intersect all link-hull halfspace constraints with the one-dimensional interpolation interval for each consecutive original approach offset. No temporal discretization. Only outside results certify absence of material; envelope inclusion is inconclusive.',
   halfspace_inflation_margins_m=[0.,.0005],by_margin_m=continuous,
   depth_quantization_boundary='0.5mm is the saved depth quantization half-step along camera z; perspective expands lateral uncertainty. This optional envelope inflation is diagnostic only and is not a new admission gate.')
  if inside:
   mat=np.stack(inside);anyinside=mat.any(0);mesh_result['by_scene_label']={}
   for name,mask in [('all',np.ones(len(hit),bool)),('table',data['table'][hit]),('object',data['obj'][hit])]:
    mesh_result['by_scene_label'][name]=dict(total=int(mask.sum()),inside_endpoint_mesh=int((mask&mat[0]).sum()),
     inside_any_discrete_sweep_pose=int((mask&anyinside).sum()),inside_only_backward_poses=int((mask&anyinside&~mat[0]).sum()),
     outside_all_discrete_triangle_meshes=int((mask&~anyinside).sum()),
     outside_all_and_surface_distance_over_half_mm=int((mask&~anyinside&(d.min(0)>.0005)).sum()),
     outside_all_surface_distance_m=summary(d.min(0)[mask&~anyinside]))
  vertices=np.asarray(asset['mesh'].vertices)@R.T+t+depth_tf[:3,3];wv=vertices@data['T'][:3,:3].T+data['T'][:3,3]
  return dict(asset_key=key,source_PCL=asset['pcl'],source_triangle_mesh=asset['source_mesh'],
   original_mask_bit_exact_replay=True,voxel_size_m=size,downsample_trace_centroids_bit_equal=exact,
   actual_backward_translation_vectors_camera_m=[a.tolist() for a in shifts],
   hit_world_z_m=summary(data['world'][hit,2]),endpoint_source_mesh_world_z_m=summary(wv[:,2]),
   original_voxel_lineage=bytype,triangle_geometry=mesh_result,
   voxel_lineage_boundary='Flags track which original endpoint/backward PCL samples contributed to each occupied query voxel through both original voxel operations. Multiple offsets may contribute to a voxel; these counts are not distinct physical contacts.')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp/outputs/dh3_real_mesh_generalization_20261103_v1'))
 parser.add_argument('--repo-root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'))
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--no-geometry-replay',action='store_true')
 args=parser.parse_args();start=time.monotonic();out=args.output_dir.resolve()
 require(not out.exists(),'Use a new diagnostic directory');out.mkdir(parents=True)
 names=['mug_yaw090','bowl_yaw000','bowl_yaw090'];reports=[];geo=None
 if not args.no_geometry_replay:geo=Geometry(args.repo_root)
 for name in names:
  scene_dir=args.root/'input_nonapple_v2/scenes'/name;run_dir=args.root/'run_nonapple_v2/inference'/name
  scene=read(scene_dir/'scene_manifest.json');collision=read(run_dir/'collision/RESULTS.json')
  data=classify(scene,scene_dir,run_dir);require(sha(run_dir/'frame/scene_points_down.npy')==collision['input']['scene_file_sha256'],'Original scene hash')
  c=data['c'];counts=data['counts'];scores=c['DH3_grasp_group_array'][:,0]
  top=int(np.argsort(scores)[::-1][0]);least=int(np.lexsort((-scores,counts.sum(1)))[0])
  rows=[candidate(i,data) for i in range(len(scores))]
  selections={}
  for tag,idx in [('highest_original_score',top),('fewest_total_overlaps_tie_highest_score',least)]:
   selections[tag]=dict(candidate=rows[idx])
   if geo:selections[tag]['geometry_diagnosis']=geo.diagnose(idx,data,collision)
  result=dict(scene_id=name,input_binding=data['binding'],original_collision_receipt=rec(run_dir/'collision/RESULTS.json'),
   original_parameters=collision['original_parameters'],original_candidate_count=len(rows),
   all_saved_candidates_rejected=bool((~c['admission_mask']).all()),
   overlap_origin_summary=dict(table_only=int(((counts[:,0]>0)&(counts[:,1]==0)).sum()),
    object_only=int(((counts[:,0]==0)&(counts[:,1]>0)).sum()),both=int(((counts[:,0]>0)&(counts[:,1]>0)).sum()),
    min_table_overlaps=int(counts[:,0].min()),min_object_overlaps=int(counts[:,1].min()),min_total_overlaps=int(counts.sum(1).min())),
   candidates=rows,selected_diagnostic_candidates=selections)
  (out/(name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  np.savez_compressed(out/(name+'_point_attribution.npz'),points_camera_m=data['points'],points_world_m=data['world'],
   source_uv=data['uv'],renderer_body_id=data['labels'],candidate_table_and_object_counts=counts,
   original_saved_masks_packed=c['scene_inclusion_masks_packed'])
  reports.append(dict(scene_id=name,receipt=rec(out/(name+'.json')),origin_summary=result['overlap_origin_summary'],
   highest_score=rows[top],fewest_overlap=rows[least]));print(json.dumps(reports[-1]),flush=True)
 source=args.repo_root/'utils/collision_detector.py';generator=args.repo_root/'generate_mesh_and_pointcloud/recover_dh3_hand_to_stl.py'
 report=dict(status='saved_collision_rejection_diagnosis_complete',scenes=reports,source=rec(__file__),
  author_collision_source=rec(source),author_PCL_generator_source=rec(generator),
  source_semantics=dict(depth='DH3 full insertion depth inherits two-finger depth and adds the learned 0/10/20/30mm; DEFAULT_DEPTH=0. The mapped source-hand PCL is translated once by +full_depth along two-finger R[:,0].',
   approach='Original range(2,int(.05*100)+1,3) executes 2cm then5cm, copying the accumulated cloud. Thus occupied source copies are at nominal0,−2,−5,−7cm along approach, not only0/−2/−5 and not a continuous swept mesh.',
   hand_model='Original300k sampled surface points, estimated normals and15 normal-offset layers through11.25mm, saved at3mm. Normal estimation does not explicitly orient normals to an inside reference.',
   gate='Scene_point inclusion in hand/approach occupied3mm voxel cells; one included visible point rejects. This is not triangle intersection, force closure, a dense scene-volume collision or a physical outcome.'),
  additional_network_forwards=0,additional_physical_steps=0,additional_renders=0,
  diagnostic_CPU_voxel_queries=geo.cpu_grid_queries if geo else 0,
  diagnostic_CPU_triangle_distance_queries=geo.triangle_distance_queries if geo else 0,
  diagnostic_CPU_closed_link_occupancy_queries=geo.occupancy_queries if geo else 0,
  author_gate_modified=False,production_artifacts_modified=False,elapsed_seconds=time.monotonic()-start,
  boundary='Post-hoc visible-depth diagnosis of three original rejection sets. Segmentation and source meshes are diagnostic inputs only. No candidate is executed or promoted, no original success-rate numerator/denominator changes.')
 (out/'RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=report['status'],output=str(out/'RESULTS.json'),elapsed_seconds=report['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()
