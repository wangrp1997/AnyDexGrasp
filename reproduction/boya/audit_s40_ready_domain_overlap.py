"""Read only DOMAIN/ready bindings, never execution labels or training records."""
from pathlib import Path
from hashlib import sha256
import collections,json,sys
import numpy as np
sys.path.insert(0,'/tmp')
from boya_prehead_domain import apply_boya_domain
ROOT=Path('/home/wangrenpeng/AnyDexGrasp');DIAG=ROOT/'outputs/s40_domain_diagnostic_20261107_v1'
QUEUE=ROOT/'outputs/s40_stable_supplement_20261107_v2/collection_resume_v3/planning/queue_manifest.json'
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def arrayref(a):
 a=np.ascontiguousarray(a);return dict(shape=list(a.shape),dtype=a.dtype.str,c_order_bytes_sha256=sha256(a.tobytes()).hexdigest())
queue=json.loads(QUEUE.read_text());identities={p.parent.name for p in DIAG.glob('graspnet_*/DOMAIN.json')};assert len(identities)==8
rows=[];bound=[];sources={str(QUEUE):sha(QUEUE)}
for oid in sorted(identities):
 domain_path=DIAG/oid/'DOMAIN.json';domain=json.loads(domain_path.read_text());sources[str(domain_path)]=sha(domain_path)
 provenance=domain['source_provenance'];source=Path(provenance['raw_npz']['path']);obs_path=Path(provenance['observation']['path'])
 for p,digest in [(source,provenance['raw_npz']['sha256']),(obs_path,provenance['observation']['sha256'])]:assert sha(p)==digest;sources[str(p)]=digest
 observation=json.loads(obs_path.read_text())
 with np.load(source,allow_pickle=False)as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
 mask=apply_boya_domain(raw,observation['T_WC'],domain['config'],source_provenance=provenance)
 assert mask.receipt['counts']==domain['counts']
 assert mask.keep_indices.tolist()==domain['kept_source_indices']
 assert mask.receipt['raw_input']==domain['raw_input']
 ready=[r for r in queue['trials']if r['object_id']==oid]
 rrows=[]
 for row in ready:
  bp=Path(row['binding_json']);assert sha(bp)==row['binding_sha256'];sources[str(bp)]=sha(bp);b=json.loads(bp.read_text())
  assert Path(b['source_npz'])==source and b['source_npz_sha256']==sha(source)
  idx=b['source_row_index'];assert type(idx)is int and 0<=idx<len(raw)
  assert np.array_equal(np.asarray(b['source_row15'],dtype=raw.dtype),raw[idx])
  assert b['source_row15_identity']==arrayref(raw[idx]);assert b['candidate_features_raw2261_identity']==arrayref(features[idx])
  assert b['trial_binding_sha256']==row['trial_binding_sha256']
  flip=b['if_flip'];assert type(flip)is bool
  used=raw[idx].copy()
  if flip:used[3:12].reshape(3,3)[:,1:3]*=-1
  np.testing.assert_array_equal(used[3:15],b['executed_camera_cgr_after_flip'])
  used_result=apply_boya_domain(used[None],observation['T_WC'],domain['config'],source_provenance=provenance)
  assert used_result.combined_mask[0]==mask.combined_mask[idx]
  depth=b['multifinger_pose_depth_type'];assert type(depth)is int and 0<=depth<4 and depth==row['extra_depth_index']
  assert b['extra_depth_m']==.01*depth and b['raw_depth_m']==float(raw[idx,2])
  assert b['multifinger_pose_finger_type']==row['gesture_id']
  record=dict(trial_id=row['trial_id'],object_id=oid,source_row_index=idx,gesture_id=row['gesture_id'],if_flip=flip,official_policy_if_flip=b['official_policy_if_flip'],extra_depth_index=depth,raw_depth_index=b['two_fingers_pose_depth_type'],in_frozen_domain=bool(mask.combined_mask[idx]),gates={name:bool(values[idx])for name,values in mask.gate_masks.items()},source_npz_sha256=b['source_npz_sha256'],binding_sha256=sha(bp),flip_preserves_domain=True)
  rrows.append(record);bound.append(record)
 c=domain['counts'];w=c['raw_width_pass'];ww=c['after_width_workspace'];j=c['joint_pass']
 rows.append(dict(object_id=oid,raw_domain_counts=c,raw_individual_rejections={k:c['input']-c[k+'_pass']for k in ['raw_width','world_workspace','world_approach']},raw_sequential_rejections=dict(raw_width=c['input']-w,world_workspace_after_width=w-ww,world_approach_after_width_workspace=ww-j),ready_count=len(ready),ready_in_domain=sum(r['in_frozen_domain']for r in rrows),ready_gate_failures={name:sum(not r['gates'][name]for r in rrows)for name in mask.gate_masks},ready_unique_raw_indices=len({r['source_row_index']for r in rrows}),ready_in_domain_unique_raw_indices=len({r['source_row_index']for r in rrows if r['in_frozen_domain']})))
sums={k:sum(r['raw_domain_counts'][k]for r in rows)for k in rows[0]['raw_domain_counts']}
report=dict(schema='anydex_s40_ready_domain_overlap_v1',scope='Planning-ready bindings only; no execution labels, evaluation files or training records read; no policy optimization',source_queue=str(QUEUE),domain_counts_totals=sums,domain_joint_candidates=sum(r['raw_domain_counts']['joint_pass']for r in rows),ready_total=len(bound),ready_in_domain=sum(r['in_frozen_domain']for r in bound),ready_gate_failures={name:sum(not r['gates'][name]for r in bound)for name in ['raw_width','world_workspace','world_approach']},objects=rows,ready_bindings=bound,input_sha256=sources,physics_steps=0,model_forwards=0,labels_read=False,policy_changed=False)
p=Path('/tmp/S40_READY_DOMAIN_OVERLAP_20261107.json');p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('ready_bindings','input_sha256')},indent=2))
