from pathlib import Path
import json,hashlib,numpy as np
r=Path('/home/wangrenpeng/AnyDexGrasp');a=r/'outputs/boya_author_pipeline_fourtype_20261114_v1';b=r/'outputs/boya_author_pipeline_fourtype_20261114_v2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(a/'LAUNCH_SOURCE_SHA256.json')['sources'];new=read(b/'LAUNCH_SOURCE_SHA256.json')['sources'];changed=[p for p in old if old[p]!=new[p]];assert changed==[str(r/'reproduction/boya/author_pipeline.py')]
rows=[]
for g in [1,6,7,8]:
 x=a/f'prepare_g{g}';y=b/f'prepare_g{g}';c=read(x/'pool/selected_author_candidate.json');d=read(y/'pool/selected_author_candidate.json')
 assert [k for k in c if c[k]!=d[k]]==['input_bindings','candidate_sha256']
 assert [k for k in c['input_bindings'] if c['input_bindings'][k]!=d['input_bindings'][k]]==['protocol']
 assert c['input_bindings']['protocol']['sha256']==d['input_bindings']['protocol']['sha256']
 with np.load(x/'pool/candidates.npz',allow_pickle=False) as p,np.load(y/'pool/candidates.npz',allow_pickle=False) as q:
  assert set(p.files)==set(q.files);same={k:p[k].shape==q[k].shape and p[k].dtype==q[k].dtype and p[k].tobytes()==q[k].tobytes() for k in p.files}
  assert [k for k in same if not same[k]]==['collection_candidate_sha256']
 oldplan=read(x/'arm/native/plan.json');newplan=read(y/'arm/native/plan.json');exclude={'output_dir','candidates','world_object_pose','process_start_monotonic_s','pregrasp_distance_m'}
 assert set(oldplan['arguments'])==set(newplan['arguments'])
 for k in oldplan['arguments']:
  if k not in exclude:assert oldplan['arguments'][k]==newplan['arguments'][k],(g,k)
 assert oldplan['arguments']['pregrasp_distance_m']==.1 and newplan['arguments']['pregrasp_distance_m']==.04
 with np.load(x/'arm/native/commands.npz',allow_pickle=False) as p,np.load(y/'arm/native/commands.npz',allow_pickle=False) as q:
  for k in ['joint_names','initial_position','phase','time_s']:assert np.array_equal(p[k],q[k]),(g,k)
  names=list(p['joint_names']);indexes=[names.index(n) for n in oldplan['hand_motor_names']]
  assert np.array_equal(p['joint_target'][:,indexes],q['joint_target'][:,indexes])
  palm_error=float(np.max(abs(p['desired_palm'][-1]-q['desired_palm'][-1])));assert palm_error<=1e-15
 rows.append(dict(gesture_id=g,candidate_numerics_all_equal=True,candidate_metadata_difference='Only COLLECTION_PROTOCOL absolute output path; protocol bytes/SHA equal, candidate SHA consequently changes.',old_candidate_sha256=c['candidate_sha256'],new_candidate_sha256=d['candidate_sha256'],candidate_pool_array_count=len(same),all_pool_numeric_and_feature_arrays_shape_dtype_bytes_equal=True,pool_only_changed_metadata_array='collection_candidate_sha256',all_hand_commands_equal=True,phases_initial_names_times_equal=True,final_desired_palm_max_matrix_error=palm_error,native_parameter_values_equal_except_pregrasp_and_new_output_paths=True,old_pregrasp_distance_m=.1,new_pregrasp_distance_m=.04))
v=dict(schema='anydex_boya_author_source4cm_controlled_change_audit_v1',status='passed',only_changed_runtime_source=changed,unchanged_transport_sha256=new[str(r/'reproduction/boya/author_transport.py')],results=rows,physics_steps=0,new_planning_calls=0,known_release_GT_defect_unchanged=True)
(b/'CONTROLLED_CHANGE_AUDIT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
