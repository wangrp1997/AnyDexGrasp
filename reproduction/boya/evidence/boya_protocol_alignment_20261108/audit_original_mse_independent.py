from pathlib import Path
import ast,copy,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');sys.path.insert(0,str(ROOT/'reproduction/boya'))
import train_boya_decision as official
R=ROOT/'outputs/boya_original_recipe_alignment_20261108_v1';P=R/'physical_control';D=R/'diagnostic'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return official.sha(p)
def zread(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def ref(p):return dict(path=str(p),sha256=sha(p))
protocol=read(P/'protocol.json');summary=read(P/'RESULTS.json');diagnostic=read(D/'RESULTS.json')
assert summary['status']=='complete'and not summary['errors']and summary['actual_grasps']==4 and summary['protocol_sha256']==sha(P/'protocol.json')
assert protocol['head_recipe']==diagnostic['head_recipe']=='code100_original_selected_mse'and protocol['physical_attempts']==4
assert protocol['outcome_dependent_retry']is False and protocol['training_writeback']is False and protocol['models_updated']is False and protocol['new_CGR_forwards']==0
frozen={}
for doc in [protocol,diagnostic]:
 for section in ['source_sha256','input_sha256']:
  for f,h in doc[section].items():
   assert f not in frozen or frozen[f]==h;frozen[f]=h
for f,h in frozen.items():assert sha(f)==h,f
training=read(ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json');assert sha(ROOT/'outputs/s40_code_recipe_20261107_v1/TRAINING_RESULTS.json')=='447302d26c1e5cccc10f23ae897105a5d845abe661640cfc4ea5fa6d912ccac0'
cls=official.official_components()[0];models={};torch.set_num_threads(4)
for g in [1,6,7,8]:
 e=training['internal_heads'][str(g)];assert sha(e['path'])==e['sha256'];c=torch.load(e['path'],map_location='cpu',weights_only=True)
 assert 'training_variant'not in c and c['epoch']==100 and c['output_dim']==20
 m=cls(**c['architecture_kwargs']);m.load_state_dict(c['model_state_dict'],strict=True);m.eval();models[g]=m
canonical=official.extract(ROOT/'robot_dh3.py',['get_graspgroup_features'],dict(np=np,copy=copy))['get_graspgroup_features']
tree=ast.parse((ROOT/'reproduction/boya/run_boya_floating_diagnostic.py').read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');loop=next(n for n in ast.walk(run)if isinstance(n,ast.For)and any(isinstance(a,ast.Call)and isinstance(a.func,ast.Attribute)and a.func.attr=='step'for a in ast.walk(n)))
loopcalls=sorted({n.func.attr for n in ast.walk(loop)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)});assert not any(any(k in s for k in ['write_root','write_joint_state','set_world_pose','set_local_pose','set_kinematic'])for s in loopcalls)
report=dict(schema='anydex_boya_original_recipe_physical_independent_audit_v1',protocol=ref(P/'protocol.json'),summary=ref(P/'RESULTS.json'),scope='Read-only known-object cached-score/control audit; no inference training or physical execution. Head forward recomputation only.',all_frozen_unique_files=len(frozen),all_frozen_hashes_match=True,physical_loop_state_setters_absent=True,physical_loop_calls=loopcalls,objects=[])
for oid in protocol['object_ids']:
 job=read(P/'jobs'/f'{oid}.json');row=job['object_row'];geo=read(D/oid/'geometry.json');g=read(Path(job['candidate_dir'])/'generation.json');record=g['records'][0]
 assert job['candidate_rank']==geo['admitted'][0]['global_score_rank']==record['global_score_rank'];assert geo['admitted'][0]['candidate_dir']==job['candidate_dir']
 first=next(x for x in geo['checked']if x['geometry']['ok']);assert first==record
 shortlist=read(D/oid/'shortlist.json')['candidates'];scores=[x['learned_score']for x in shortlist];assert all(a>=b for a,b in zip(scores,scores[1:]));assert not any(any(k in x for k in ['label','success','result','trial_outcome'])for x in shortlist)
 q=zread(D/oid/'head_outputs.npz');obs=read(row['observation_json']);raw=zread(Path(row['representation_dir'])/'raw_camera_candidates.npz');rawfeatures=raw['candidate_features']
 cgr=canonical(np.c_[rawfeatures[q['source_indices']].copy(),q['if_flip']],None)['grasp_preds_features'];recomputed=np.empty_like(q['outputs20'])
 for gesture,model in models.items():
  ix=np.flatnonzero(q['gesture_ids']==gesture)
  with torch.inference_mode():
   for start in range(0,len(ix),128):
    group=ix[start:start+128];recomputed[group]=model(torch.from_numpy(cgr[group]))[0][:,0].numpy()
 error=float(np.max(np.abs(recomputed-q['outputs20'])))if len(recomputed)else 0.;assert error<1e-6
 for candidate in shortlist:
  assert abs(float(recomputed[candidate['query_index'],candidate['target_index']])-candidate['learned_score'])<1e-6
 E=P/'execution'/oid;e=read(E/'evaluation.json');native=read(E/'native_audit.json');t=zread(E/'trace.npz');commands=zread(E/'commands.npz');pool=zread(Path(job['candidate_dir'])/'candidates.npz');n=e['total_physics_steps']
 assert e['complete']and n==len(t['time_s']) and e['arm_present']is False
 assert sha(E/'trace.npz')==e['trace_sha256']and sha(E/'commands.npz')==e['commands_sha256']
 assert sha(Path(job['candidate_dir'])/'candidates.npz')==e['hand_candidate_sha256']==g['candidates_sha256']
 assert all(e[k]==0 for k in ['state_writes_during_execution','root_pose_writes_during_execution','object_state_writes_during_execution','new_CGR_forwards','new_training_updates'])
 assert row['object_mass_kg']==.1 and row['object_friction']==.6 and abs(e['audit']['mass_kg']-.1)<2e-8 and e['object_weight_world_n']==[0.,0.,-.98]
 assert np.allclose(e['audit']['initialization']['expected_object_pose_xyzw'],row['world_pose_xyzw'],atol=2e-15,rtol=0)
 for mat in e['audit']['materials'].values():assert abs(mat['static_friction']-.6)<1e-7 and abs(mat['dynamic_friction']-.6)<1e-7
 assert all(np.isfinite(x).all()for x in t.values()if x.dtype.kind in 'fiu')
 for key in ['joint_target','velocity_target','desired_palm','phase','time_s']:assert np.array_equal(t[key],commands[key][:n]),(oid,key)
 assert np.array_equal(t['time_s'],np.arange(1,n+1)*.004)
 names=list(t['joint_names']);mi=[names.index(x)for x in pool['independent_motor_names']];opened=pool['proposal_q'][0];closed=pool['proposal_q_close'][0];palm=pool['proposal_palm'][0];direction=pool['proposal_approach_direction'][0]
 wanted=np.repeat(opened[None],2100,axis=0);v=np.arange(1,251,dtype=float)/250.;wanted[1300:1550]=(1-v[:,None])*opened+v[:,None]*closed;wanted[1550:]=closed
 assert np.array_equal(t['joint_target'][:,mi],wanted[:n].astype(np.float32))
 desired=np.repeat(palm[None],2100,axis=0);pre=palm[:3,3]-.1*direction;desired[:800,:3,3]=pre;s=np.arange(1,501)/500.;u=s*s*s*(10.+s*(-15.+6.*s));desired[800:1300,:3,3]=pre+u[:,None]*.1*direction;desired[1600:,:3,3]+=u[:,None]*[0.,0.,.1]
 assert np.allclose(t['desired_palm'],desired[:n],atol=2e-14,rtol=0)
 urdf=ET.parse(E/'boya_translation_gantry.urdf').getroot();limit=mimic=0.
 for j in urdf.findall('joint'):
  name=j.get('name')
  if name not in names:continue
  values=t['joint_pos'][:,names.index(name)];li=j.find('limit');mi_j=j.find('mimic')
  if li is not None and j.get('type')in ['revolute','prismatic']:limit=max(limit,float(np.maximum(np.maximum(float(li.get('lower'))-values,values-float(li.get('upper'))),0).max()))
  if mi_j is not None:mimic=max(mimic,float(np.abs(values-t['joint_pos'][:,names.index(mi_j.get('joint'))]*float(mi_j.get('multiplier','1'))-float(mi_j.get('offset','0'))).max()))
 assert abs(limit-e['maxima']['joint_limit_violation_rad'])<2e-7 and abs(mimic-e['maxima']['mimic_error_rad'])<2e-7
 pp=np.linalg.norm(t['palm_pose'][:,:3,3]-desired[:n,:3,3],axis=1);assert np.isclose(pp.max(),e['maxima']['palm_position_error_m'],atol=1e-12)
 penetration=float(max(t['object_penetration_m'].max(),t['robot_penetration_m'].max()));assert np.isclose(penetration,e['maxima']['penetration_m'],atol=1e-12)
 assert native['within_20um']and native['physics_steps']==0 and native['max_source_outside_cooked_bound_m']<=2e-5 and native['max_cooked_outside_source_bound_m']<=2e-5
 mesh=Path(row['mesh_path']);assert sha(mesh)==row['mesh_sha256'];vertices=np.array([[float(x)for x in l.split()[1:4]]for l in mesh.read_text().splitlines()if l.startswith('v ')])*row['scale']
 poses=t['object_pose_xyzw'].astype(float);rot=Rotation.from_quat(poses[:,3:]).as_matrix();bottom=np.empty(n)
 for i in range(0,n,64):bottom[i:i+64]=(vertices@rot[i:i+64,2,:].T+poses[i:i+64,2]).min(0)
 phases={str(ph):int(np.sum(t['phase']==ph))for ph in dict.fromkeys(t['phase'])};lift=t['phase']=='lift';hold=np.flatnonzero(t['phase']=='hold');both=(t['finger_loads_n'][:,0]>.001)&(t['finger_loads_n'][:,1:].sum(1)>.001)
 if len(hold):
  rise=poses[:,2]-poses[hold[-1],2];picked=lift&(bottom>=.005)&(rise>=.005)&both;pickup=e['tabletop_pickup'];assert int(picked.sum())==pickup['qualifying_steps']and bool(picked[-1])==pickup['final_state_still_lifted_and_bearing'];assert np.isclose(bottom[-1],pickup['final_bottom_clearance_m'],atol=2e-12)and np.isclose(rise[-1],pickup['final_reference_rise_m'],atol=2e-12)
  success=bool(all(e['gates'].values())and picked[-1]and rise[-1]>=.05)
  pickup_stats=dict(qualifying_steps=int(picked.sum()),final_bottom_clearance_m=float(bottom[-1]),peak_lift_bottom_clearance_m=float(bottom[lift].max()),final_reference_rise_m=float(rise[-1]),two_sides_loaded_lift_steps=int((both&lift).sum()),final_still_bearing=bool(picked[-1]))
 else:
  assert oid=='egad_A3'and n==1175 and e['failure_reason']=='object_pushed_before_closure'and e['tabletop_pickup']is None and not lift.any()
  assert e['gates']['complete_execution']is False and all(v for k,v in e['gates'].items()if k!='complete_execution');success=False
  pickup_stats=dict(qualifying_steps=0,final_bottom_clearance_m=float(bottom[-1]),final_reference_rise_m=None,hold_or_lift_executed=False,reason='Normal declared approach task-stop; not infrastructure failure')
 assert success==e['task_success'];report['objects'].append(dict(object_id=oid,rank=job['candidate_rank'],gesture_id=record['gesture_id'],target_index=record['target_index'],learned_score=record['learned_score'],selected_before_physics_as_first_geometry_admitted=True,independent_original_head_full_output_max_abs_error=error,all_shortlist_scores_recomputed=True,source_pose_mass_friction_collision_assets_unchanged=True,original_nominal_finger_command_path_exact_float32=True,source_desired_palm_path_reconstructed=True,steps=n,phases=phases,gates=e['gates'],failure_reason=e['failure_reason'],palm_error_m=float(pp.max()),max_penetration_m=penetration,task_success=success,recomputed_pickup=pickup_stats,final_finger_loads_n=t['finger_loads_n'][-1].tolist(),evaluation=ref(E/'evaluation.json'),trace=ref(E/'trace.npz')))
# Outcome labels are opened only after all selection/physical checks above; this support summary never feeds any ranking.
dataset=read(training['dataset_manifest']['path']);a=zread(dataset['gesture_files']['6']['path']);ix=(a['split']=='train')&(a['target_index']==12)
report['E0_selected_g6_target12_training_support']=dict(fit_count=int(ix.sum()),fit_positive=int(a['labels'][ix].sum()),fit_negative=int(np.sum(a['labels'][ix]==0)),all_records_count=int(np.sum(a['target_index']==12)),diagnostic_only_after_fixed_execution=True)
report.update(success=sum(x['task_success']for x in report['objects']),physical_steps=sum(x['steps']for x in report['objects']),physics_trials=4,independent_unseen_validation=False,new_physics_steps_by_audit=0,new_training_updates_by_audit=0,new_CGR_forwards_by_audit=0,
 limitations=['Known four scenes with existing coarse collision assets; not independent external benchmark or full original hardware reproduction.','E0 changes head/selected pose and becomes successful under same controlled-wrist environment; this does not isolate score calibration from pose/control interaction.','Full-trajectory low-level contacts are not retained; bearing recomputation uses saved per-finger normal loads and complete source vertices.','Zero state writes: bound reported counters plus direct runner loop inspection; not independent runtime API instrumentation.'])
assert report['success']==summary['success']==1 and report['physical_steps']==summary['physical_steps']==7475
out=Path('/tmp/BOYA_ORIGINAL_RECIPE_PHYSICAL_AUDIT_20261108.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['physical_loop_calls']},indent=2))
