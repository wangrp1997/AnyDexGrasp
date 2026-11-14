"""Existing inputs/FK only. No IK, least-squares, collision planning or Kit."""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize

def forbidden(*a,**k):raise AssertionError('Optimization forbidden in read-only FK audit')
scipy.optimize.least_squares=forbidden
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
CANON=Path('/home/wangrenpeng/CanonDex')
sys.path.insert(0,str(CANON))
from canondex.planning.model import PlanningModel
NEW=ROOT/'outputs/boya_full_data_lift_20261114_v1'
OLD=ROOT/'outputs/boya_author_pipeline_fourtype_20261114_v2'
OUT=Path(__file__).resolve().parent
reads={}
def read(p):
 p=Path(p).resolve();reads[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
def error(actual,target):
 return dict(position_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),rotation_rad=float(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude()))
newobs=read(NEW/'scene/capture/observation.json')
oldobs=read(ROOT/'outputs/boya_author_scene_20261114_v4/capture/observation.json')
newscene=read(newobs['scene_json']);oldscene=read(oldobs['scene_json'])
rows=[];fk_count=0
native_sources=[CANON/'scripts/research/run_template_arm_positive.py',CANON/'canondex/grasp/template_arm_admission.py',CANON/'canondex/grasp/admm/arm_only_ik.py',CANON/'canondex/grasp/template_arm_bridge.py',CANON/'canondex/planning/model.py',CANON/'canondex/robots/kinematics.py']
native_source_rows=[]
for p in native_sources:
 key=str(p);oldsha=oldscene['source_sha256'][key];newsha=newscene['source_sha256'][key]
 assert oldsha==newsha==sha(p)
 native_source_rows.append(dict(path=key,sha256=oldsha,old_new_current_identical=True))
for g in [1,6,7,8]:
 nd=NEW/f'g{g}';od=OLD/f'prepare_g{g}'
 n=read(nd/'arm/native/frozen_input.json');o=read(od/'arm/native/frozen_input.json');plan=read(od/'arm/native/plan.json')
 nc=read(nd/'pool/selected_author_candidate.json');oc=read(od/'pool/selected_author_candidate.json')
 failure=read(nd/'arm/native/admission_failure.json');receipt=read(nd/'arm/ARM_PREPARATION.json')
 excluded={'output_dir','candidates','world_object_pose','process_start_monotonic_s'}
 argdiff={k:[o['arguments'].get(k),n['arguments'].get(k)]for k in set(o['arguments'])|set(n['arguments']) if k not in excluded and o['arguments'].get(k)!=n['arguments'].get(k)}
 assert not argdiff,argdiff
 assert n['seeds']==o['seeds'] and n['initial']==o['initial']
 poolerrors={};targets={}
 for name,d,f in [('new',nd,n),('old',od,o)]:
  p=d/'pool/candidates.npz';reads[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:pool={k:z[k].copy()for k in z.files}
  world=read(d/'pool/world_object_pose.json');body=np.asarray(pool['world_T_object']);world=np.asarray(world)
  rotation=world[:3,:3]@body[:3,:3].T
  relocated=np.asarray(pool['proposal_palm'][0]).copy();relocated[:3,:3]=rotation@relocated[:3,:3];relocated[:3,3]=world[:3,3]+rotation@(relocated[:3,3]-body[:3,3])
  direction=rotation@pool['proposal_approach_direction'][0]
  wanted=np.asarray(f['palm']).copy();wanted[:3,3]-=.04*direction
  poolerrors[name]=dict(primary_source_vs_world_max_error=diff(body,world),source_palm_vs_bridge_max_error=diff(f['source_palm'],pool['proposal_palm'][0]),native_palm_vs_bridge_max_error=diff(f['palm'],pool['proposal_palm'][0]),independent_relocation_error=diff(relocated,f['palm']),normalized_direction=direction.tolist(),target_pregrasp_world=wanted.tolist())
  targets[name]=wanted
 model=PlanningModel(Path(n['arguments']['robot_urdf']));base=np.asarray(n['base_transform'])
 cmd=od/'arm/native/commands.npz';reads[str(cmd)]=sha(cmd)
 with np.load(cmd,allow_pickle=False) as z:
  last=np.where(z['phase']=='home_to_pregrasp')[0][-1];q=model.full_state(dict(zip(map(str,z['joint_names']),map(float,z['joint_target'][last]))))
 actual=model.fk.transforms(q,base)['palm'];fk_count+=1
 seedrows=[]
 for index,seed in enumerate(n['seeds']):
  palm=model.fk.transforms(dict(n['initial'],**seed),base)['palm'];fk_count+=1
  seedrows.append(dict(seed_index=index,seed_rad=seed,existing_seed_fk_vs_new_target=error(palm,targets['new']),existing_seed_fk_vs_old_target=error(palm,targets['old'])))
 oldnative=[dict(path=p,sha256=h,current_sha256=sha(p),unchanged=sha(p)==h)for p,h in plan['source_sha256'].items()]
 assert all(r['unchanged']for r in oldnative)
 for key,path in [('robot_urdf_sha256','robot_urdf'),('physics_config_sha256','physics_config'),('diagnostic_config_sha256','diagnostic_config')]:assert n[key]==o[key]==sha(n['arguments'][path])
 audit=dict(gesture_id=g,status=read(nd/'LIFT_PIPELINE_RESULTS.json')['status'],reason=failure['reason'],new_native_plan_exists=(nd/'arm/native/plan.json').exists(),
 candidate=dict(new_raw_row=nc['source_row_index'],old_raw_row=oc['source_row_index'],new_score=nc['score'],old_score=oc['score'],new_extra_m=nc['extra_depth_m'],old_extra_m=oc['extra_depth_m'],new_flip=nc['source_if_flip'],old_flip=oc['source_if_flip'],new_slot=nc['slot_id'],old_slot=oc['slot_id']),
 native_input=dict(semantic_arguments_equal=True,excluded_path_and_clock_keys=sorted(excluded),eight_seeds_equal=True,initial_full28_equal=True,base_transform_max_error=diff(n['base_transform'],o['base_transform']),opened_motors_equal=n['open_motors']==o['open_motors'],closed_motors_equal=n['closed_motors']==o['closed_motors'],native_sources=oldnative),
 palm_delta=dict(translation_m=(np.asarray(n['palm'])[:3,3]-np.asarray(o['palm'])[:3,3]).tolist(),translation_norm_m=float(np.linalg.norm(np.asarray(n['palm'])[:3,3]-np.asarray(o['palm'])[:3,3])),rotation_deg=float(np.rad2deg(Rotation.from_matrix(np.asarray(o['palm'])[:3,:3].T@np.asarray(n['palm'])[:3,:3]).magnitude())),new_world_xyz=np.asarray(n['palm'])[:3,3].tolist(),old_world_xyz=np.asarray(o['palm'])[:3,3].tolist()),
 compatibility_relocation=poolerrors,
 existing_old_pregrasp_q_fk=dict(vs_old_target=error(actual,targets['old']),vs_new_target=error(actual,targets['new']),original_selected_seed_index=plan['pregrasp_ik']['motion_seed_selection']['selected_seed_index']),
 declared_seed_fk_only=seedrows,
 original_planning_cost=dict(declared_ik_queries=8,per_query_max_nfev=300,actual_nfev_unavailable=True,wall_s=receipt['wall_s'],physics_steps=0),
 limitation='Native rejection receipt does not persist failed optimizer endpoints/nfev/residuals; this audit does not rerun any IK and cannot infer those missing quantities or global unreachability')
 rows.append(audit)
# Budget is read from original AST, rather than inferred from elapsed time.
rank=ast.parse((CANON/'canondex/grasp/template_arm_admission.py').read_text())
func=next(n for n in rank.body if isinstance(n,ast.FunctionDef)and n.name=='rank_arm_ik_solutions')
defaults={a.arg:ast.literal_eval(v)for a,v in zip(func.args.kwonlyargs,func.args.kw_defaults)if isinstance(v,ast.Constant)}
assert defaults['max_nfev']==300
sourceposes={r['object_id']:r['world_T_object']for r in oldscene['objects']}
poseerrors={r['object_id']:diff(r['world_T_object'],sourceposes[r['object_id']])for r in newscene['objects']}
for p,h in reads.items():assert sha(p)==h
result=dict(schema='anydex_boya_lift_null_readonly_audit_v1',scope='No candidate replacement/IK/optimization/planning/Kit/physics; existing source and commands/FK only',
 conclusion='Different fixed source grasp poses with unchanged native IK/initial/seeds/budget; four local misses before HOME or collision do not establish pipeline unusability or global unreachability',
 camera_world_transform_max_error=diff(newobs['T_WC'],oldobs['T_WC']),observed_source_pose_max_errors=poseerrors,native_current_capture_old_capture_sources=native_source_rows,
 per_seed_max_nfev_from_source_ast=defaults['max_nfev'],conditions=rows,input_sha256=reads,
 costs=dict(physical_steps=0,new_labels=0,new_CGR=0,new_IK_calls=0,new_planning_calls=0,existing_state_FK_evaluations=fk_count),
 original_four_condition_cost=dict(declared_ik_queries=32,actual_nfev_unavailable=True,physics_steps=0,plans=0,home_driver_calls=0,lift_preparations=0),
 next_scope='Separate frozen multi-scene collection protocol; no same-candidate extra seed/threshold change and no success-rate evaluation before data/training')
(OUT/'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(conditions=[dict(g=r['gesture_id'],translation_norm_m=r['palm_delta']['translation_norm_m'],rotation_deg=r['palm_delta']['rotation_deg'],old_q_error=r['existing_old_pregrasp_q_fk'])for r in rows],costs=result['costs'],camera_error=result['camera_world_transform_max_error'],object_pose_errors=poseerrors,native_sources_identical=len(native_source_rows)),indent=2))
