"""Read-only original DH3 collision-vs-execution insertion-depth comparison."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diag-dir',type=Path,required=True);p.add_argument('--repo-root',type=Path,default=Path('/home/wangrenpeng/AnyDexGrasp'));a=p.parse_args();source=a.repo_root/'ur_toolbox/ur_toolbox/robot/ur.py';text=source.read_text()
 assert 'back_dis = self.angle[0] / 100 * 0.005' in text
 assert '(multifinger_grasp_used.depth - back_dis) * target_gripper_pose' in text
 rows=[]
 for name in ['mug_yaw090','bowl_yaw000','bowl_yaw090']:
  r=json.loads((a.diag_dir/(name+'.json')).read_text())
  for key,x in r['selected_diagnostic_candidates'].items():
   c=x['candidate'];back=c['position_register']/100*.005;z=x['geometry_diagnosis']['endpoint_source_mesh_world_z_m']['min'];worlddir=c['approach_unit_world']
   rows.append(dict(scene=name,candidate_label=key,candidate_index=c['index'],original_collision_full_depth_m=c['full_DH3_depth_m'],
    original_DH3_execution_backoff_m=back,original_execution_effective_depth_m=c['full_DH3_depth_m']-back,
    collision_mesh_min_world_z_m=z,analytically_depth_corrected_source_mesh_min_world_z_m=z-back*worlddir[2]))
 out=a.diag_dir/'DEPTH_SEMANTICS.json';assert not out.exists()
 result=dict(status='original_depth_semantics_compared',source=rec(__file__),original_execution_source=rec(source),
  author_collision_source=rec(a.repo_root/'utils/collision_detector.py'),rows=rows,
  finding='Author collision uses full DH3 depth, while author DH3 execution subtracts position_register/100*5mm. Thus the original collision proxy is roughly4-4.75mm deeper than the corresponding original executed source-hand pose for these six examples. This discrepancy exists in original code and was retained in the faithful reproduction.',
  boundary='Analytic rigid translation of saved source meshes only. These are not executed poses or observed physical outcomes; no gate is changed or bypassed.',
  additional_network_forwards=0,additional_physical_steps=0,additional_collision_queries=0)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(out),sha256=sha(out),rows=rows),indent=2))
if __name__=='__main__':main()
