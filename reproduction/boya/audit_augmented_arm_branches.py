"""Independently invert original branch transforms and verify every saved row."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2]/'outputs/boya_augmented_arm_external6_20261114_v1'
out=[]
for oid in tuple('egad_'+r+str(c) for r in 'ACE' for c in (2,5)):
 d=R/oid/'representation';r=json.loads((d/'RESULTS.json').read_text());assert r['status']=='complete'
 assert r['network_forward_count']==11 and r['model_load_count']==1
 assert r['identity_regression']['all_arrays_exactly_equal']
 pmax=rmax=0.;offset=0
 with np.load(d/'raw_camera_candidates.npz',allow_pickle=False) as z:
  merged={k:z[k].copy() for k in z.files}
 for branch in r['branches']:
  files=branch['output_files']
  for f in files.values():assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
  with np.load(files['raw_camera_candidates.npz']['path']) as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
  with np.load(files['recovered_camera_candidates.npz']['path']) as z:recovered=z['candidates'].copy();rf=z['candidate_features'].copy()
  inv=np.linalg.inv(branch['transform']);expected_p=raw[:,12:]@inv[:3,:3].T+inv[:3,3];expected_r=inv[:3,:3]@raw[:,3:12].reshape(-1,3,3)
  if branch['scene_reflection']:expected_r[:,:,1]*=-1
  pmax=max(pmax,float(abs(expected_p-recovered[:,12:]).max()));rmax=max(rmax,float(abs(expected_r-recovered[:,3:12].reshape(-1,3,3)).max()))
  n=len(raw);end=offset+n
  np.testing.assert_array_equal(features,rf);np.testing.assert_array_equal(merged['candidate_features'][offset:end],features)
  np.testing.assert_array_equal(merged['candidates'][offset:end],recovered)
  np.testing.assert_array_equal(merged['source_branch'][offset:end],np.full(n,branch['index']))
  np.testing.assert_array_equal(merged['source_branch_row'][offset:end],np.arange(n))
  np.testing.assert_array_equal(merged['source_scene_reflection'][offset:end],np.full(n,branch['scene_reflection']))
  offset=end
 assert pmax<2e-7 and rmax<5e-7 and offset==22528
 protocol=json.loads((R/oid/'REPRESENTATION_PROTOCOL.json').read_text());ref=Path(protocol['identity_reference_dir'])
 assert (d/'representation_outputs.npz').read_bytes()==(ref/'representation_outputs.npz').read_bytes()
 row=dict(object_id=oid,rows=offset,feature_pairing_exact=True,identity_exact=True,collision_cloud_exact=True,max_inverse_position_error_m=pmax,max_inverse_rotation_element_error=rmax)
 out.append(row);print(row,flush=True)
json.dump(dict(status='complete',cases=out,new_models_loaded=0,new_forwards=0,new_physics_steps=0,audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),open(R/'INDEPENDENT_BRANCH_AUDIT.json','w'),indent=2)
