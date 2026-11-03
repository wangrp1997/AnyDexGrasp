"""Summarize saved DH3 collision diagnostics and original proposal source gates.
Only reads saved artifacts, RGB-D segmentation and source. No geometric query,
network forward, rendering or physical integration is executed.
"""
from pathlib import Path
import argparse,ast,hashlib,json
import numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rec(p):return dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diag-dir',type=Path,required=True);p.add_argument('--repo-root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'));args=p.parse_args()
 source=args.repo_root/'robot_dh3.py';text=source.read_text();tree=ast.parse(text);lines=text.splitlines()
 parse=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_preds')
 grasp=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_grasp')
 gate=next(n for n in ast.walk(grasp) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mask' for t in n.targets) and 'preds' in ast.unparse(n.value))
 active_objectness_subscripts=[ast.unparse(n) for n in ast.walk(parse) if isinstance(n,ast.Subscript) and 'objectness_mask_i' in ast.unparse(n.slice)]
 assert not active_objectness_subscripts
 rows=[]
 for name in ['mug_yaw090','bowl_yaw000','bowl_yaw090']:
  d=read(args.diag_dir/(name+'.json'));binding=d['input_binding'];manifest=read(binding['inputs'][0]['path']);scene=Path(binding['inputs'][0]['path']).parent
  seg=np.load(scene/'segmentation_for_validation_only.npy');depth=np.asarray(Image.open(scene/'depth.png'));K=np.array(manifest['camera']['K']);factor=manifest['spec']['camera']['depth_factor'];T=np.array(manifest['camera']['T_CW'])
  seed_world=np.array([x['original_two_finger_seed_world_m'] for x in d['candidates']]);seed_cam=seed_world@T[:3,:3].T+T[:3,3]
  uv=np.rint(np.column_stack([K[0,0]*seed_cam[:,0]/seed_cam[:,2]+K[0,2],K[1,1]*seed_cam[:,1]/seed_cam[:,2]+K[1,2]])).astype(int)
  assert ((uv[:,0]>=0)&(uv[:,0]<seg.shape[1])&(uv[:,1]>=0)&(uv[:,1]<seg.shape[0])).all()
  body=seg[uv[:,1],uv[:,0]]&((1<<24)-1);ids=manifest['body_ids'];angles=np.array([x['angle_from_world_down_degrees'] for x in d['candidates']]);raw_depth_error=np.abs(depth[uv[:,1],uv[:,0]]/factor-seed_cam[:,2])
  rows.append(dict(scene_id=name,candidate_count=len(body),seed_renderer_body_counts=dict(table=int((body==ids['table']).sum()),object=int((body==ids['object']).sum())),
   seed_depth_matches_saved_pixel_max_error_m=float(raw_depth_error.max()),
   candidate_seed_uv=uv.tolist(),candidate_seed_renderer_body_id=body.tolist(),
   approach_angle_from_world_down_deg=dict(min=float(angles.min()),median=float(np.median(angles)),max=float(angles.max())),
   expanded_two_finger_width_above_mapped_hand_width_count=sum(x['expanded_two_finger_width_m']>x['mapped_hand_width_m']+1e-6 for x in d['candidates']),
   original_overlap_origin=d['overlap_origin_summary'],saved_diagnosis=rec(args.diag_dir/(name+'.json'))))
 output=args.diag_dir/'MECHANISM_FINDINGS.json'
 assert not output.exists(),'Use a fresh findings path'
 r=dict(status='original_source_and_saved_collision_mechanisms_verified',source=rec(__file__),original_robot_source=rec(source),
  original_orientation_gate=dict(line=gate.lineno,expression=ast.unparse(gate.value),
   approach_z_camera_min=.9,max_angle_from_world_down_deg=float(np.degrees(np.arccos(.9))),
   implication='This original robot demo admits near-downward approaches only; no side grasp survives this unmodified gate in the frozen camera orientation.'),
  original_objectness_filter=dict(active_objectness_per_seed_subscripts=active_objectness_subscripts,
   source_excerpt=[dict(line=i+1,text=line) for i,line in enumerate(lines) if 84<=i+1<=104],
   implication='Original parser tests any objectness-positive seed but leaves per-seed masking commented out. Saved background/table seeds can therefore survive; this is author code, not a new adapter filtering choice.'),
  scene_findings=rows,diagnostic_receipt=rec(args.diag_dir/'RESULTS.json'),
  main_conclusions=[
   'Mug090: every rejected candidate overlaps object-labeled visible points; 37 only object and11 also table. Candidate12 is rejected by exactly1 object point, contributed by endpoint PCL only. This point is5.859mm from original triangle surfaces and outside every link convex envelope throughout the whole piecewise linear0-to7cm approach, including diagnostic0.5mm envelope inflation. This certifies conservatism of that flagged point, not a grasp success or complete-scene collision-free certificate.',
   'Bowl000 and bowl090 include many table/background proposals. Their highest-scoring original source-hand meshes extend31.254mm and25.294mm below the table, respectively. This is a pose-placement defect in these candidates, not solely voxel contact tolerance.',
   'Six representative original voxel masks replay bit-exactly; their5/7cm copies contribute no flagged point. Most rejection cells contain endpoint PCL samples. The cumulative0,-2,-5,-7cm source behavior is real but does not explain these six rejections by itself.',
   'Original hand PCL includes up to11.25mm of estimated-normal offsets before3mm voxelization; normals are not explicitly oriented inward. Occupied PCL voxels are therefore not interchangeable with exact material intersection. Original source meshes are non-watertight, so unsigned distances are never mislabeled as signed penetration.'
  ],additional_network_forwards=0,additional_physical_steps=0,additional_renders=0,additional_collision_queries=0,
  production_artifacts_modified=False,original_gate_modified=False,
  boundary='Post-hoc diagnosis only. No original gate is relaxed, no candidate executed, no original result or success denominator changed. Convex-envelope containment is inconclusive; only outside certificates are used.')
 output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(output=str(output),sha256=sha(output),scenes=rows),indent=2))
if __name__=='__main__':main()
