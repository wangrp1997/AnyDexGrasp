"""Static, label-free alternative CGR datum; never deployment with old head.
Fixed 65 samples of each saved q_open->q_close; find closest thumb-index pad
separation to original raw jaw width (not x1.7). Build contact frame using the
exact asset_adapter.calibrate convention, then rerun original OPEN-hand geometry.
No full object mesh/identity-based rules/physics/head inference/outcomes used.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/wangrenpeng/AnyDexGrasp/reproduction/boya')
from asset_adapter import BoyaAssets,cgr_reference_pose,map_slot,rigid
from build_collection_pool import OpeningGeometryCache,geometry
ROOT=Path('/home/wangrenpeng/AnyDexGrasp')
SAMPLES=65
CASES=[('008','s40_hand_domain_20261107_v1',2),('021','s40_hand_domain_20261107_v1',0),('044_new','s40_hand_domain_20261107_v1',7),('044_old','s40_hand_domain_20261107_v1',10),('066','s40_hand_domain_20261107_v1',1),('069','s40_hand_domain_20261107_v1',0)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={}
def rd(p):
 p=Path(p);inputs[str(p.resolve())]=sha(p);return json.loads(p.read_text())
def npload(p):
 p=Path(p);inputs[str(p.resolve())]=sha(p);return np.load(p,allow_pickle=False)
def contact_frame(a,q):
 c=[a.reference_contacts[g]for g in ('th','ff')]
 points,normals=a.hand.evaluate(q,[x['collision_link']for x in c],[x['point_link_m']for x in c],[x['normal_link']for x in c],jacobian=False)[:2]
 n=normals[0]/np.linalg.norm(normals[0]);approach=np.array([0.,1.,0.]);approach-=(approach@n)*n
 if np.linalg.norm(approach)<1e-6:
  approach=np.eye(3)[int(np.argmin(abs(n)))];approach-=(approach@n)*n
 approach/=np.linalg.norm(approach)
 delta=points[1]-points[0];width=float(np.linalg.norm(delta));x=delta/width
 z=approach-(approach@x)*x;z/=np.linalg.norm(z)
 frame=np.eye(4);frame[:3,:3]=np.column_stack((x,np.cross(z,x),z));frame[:3,3]=points.mean(0)
 return width,rigid(frame)
def angle(R):return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
start=time.monotonic();a=BoyaAssets();cache=OpeningGeometryCache();cal=rd(ROOT/'outputs/boya_adapter_preparation_20261105_v2/calibration.json');slots={s['slot_id']:s for s in cal['slots']};result=[]
for label,run,rank in CASES:
 oid='graspnet_'+label[:3];pool=ROOT/'outputs'/run/oid/f'pool_{rank:05d}'
 generation=rd(pool/'generation.json');row=generation['records'][0];observation=rd(generation['source_observation']);camera=np.array(observation['T_WC'])
 with npload(Path(generation['source_frame'])/'raw_camera_candidates.npz') as z:raw=z['candidates'][row['source_candidate_index']].copy()
 with npload(Path(generation['source_frame'])/'representation_outputs.npz') as z:cloud=z['sparse_input_XYZ_features'].copy()
 cloud=cloud@camera[:3,:3].T+camera[:3,3]
 with npload(pool/'candidates.npz') as z:q0=z['proposal_q'][0].copy();q1=z['proposal_q_close'][0].copy();savedpalm=z['proposal_palm'][0].copy();savedapproach=z['proposal_approach_direction'][0].copy()
 slot=slots[row['source_slot']];used=raw.copy()
 if row['if_flip']:used[3:12].reshape(3,3)[:,1:3]*=-1
 insertion=.01*row['extra_depth_index'];oldpalm=map_slot(used,camera,slot,insertion_m=insertion);approach=cgr_reference_pose(used,camera)[:3,2]
 if not np.allclose(oldpalm,savedpalm,rtol=0,atol=1e-12)or not np.allclose(approach,savedapproach,rtol=0,atol=1e-12):raise ValueError('Original mapping reconstruction mismatch')
 records=[]
 for i,f in enumerate(np.linspace(0.,1.,SAMPLES)):
  q=(1-f)*q0+f*q1;width,frame=contact_frame(a,q)
  records.append(dict(sample=i,fraction=float(f),width_m=width,absolute_width_error_m=abs(width-float(raw[1])),palm_T_reference=frame.tolist(),q=q.tolist()))
 selected=min(records,key=lambda x:(x['absolute_width_error_m'],x['sample']))
 # Only datum changes; exact original saved q_open/q_close are retained.
 adapted_slot=dict(slot,palm_T_reference=selected['palm_T_reference'])
 newpalm=map_slot(used,camera,adapted_slot,insertion_m=insertion)
 oldcheck=geometry(a,q0,oldpalm,cloud,approach,cache);newcheck=geometry(a,q0,newpalm,cloud,approach,cache)
 delta=np.linalg.inv(oldpalm)@newpalm
 out=dict(case_label=label,object_id=oid,source_candidate_index=row['source_candidate_index'],gesture_id=row['gesture_id'],source_slot=row['source_slot'],if_flip=row['if_flip'],extra_depth_index=row['extra_depth_index'],original_raw_width_m=float(raw[1]),original_open_reference_width_m=slot['reference_width_m'],selected_contact_state=selected,palm_translation_delta_world_m=(newpalm[:3,3]-oldpalm[:3,3]).tolist(),palm_translation_delta_norm_m=float(np.linalg.norm(newpalm[:3,3]-oldpalm[:3,3])),palm_rotation_delta_rad=angle(delta[:3,:3]),old_geometry=oldcheck,new_geometry=newcheck,old_palm=oldpalm.tolist(),new_palm=newpalm.tolist(),q_open_unchanged=True,q_close_unchanged=True,approach_unchanged=True,width_curve_m=[x['width_m']for x in records])
 result.append(out)
 print(label,'raw',round(float(raw[1])*1e3,3),'fraction',selected['fraction'],'errmm',round(selected['absolute_width_error_m']*1e3,3),'delta mm',round(out['palm_translation_delta_norm_m']*1000,3),'rotdeg',round(np.degrees(out['palm_rotation_delta_rad']),3),'old',oldcheck,'new',newcheck,flush=True)
report=dict(schema='anydex_boya_contact_width_reference_static_probe_v1',policy=dict(sample_count=SAMPLES,selection='minimum absolute thumb/index annotated pad separation error against ORIGINAL raw jaw width; ties first sample',frame='asset_adapter.calibrate thumb-normal tangent projection exact formula',retained='Original open_q, close_q, flip, raw depth, additional insertion, world approach, geometry gate',new_mapping_requires_new_labels_or_explicit_transfer_validation=True,old_head_considered_adapted_to_new_mapping=False,labels_or_outcomes_read=False,full_object_geometry_read=False),physics_steps=0,training_steps=0,network_forwards=0,cases=result,old_admitted=sum(x['old_geometry']['ok']for x in result),new_admitted=sum(x['new_geometry']['ok']for x in result),wall_s=time.monotonic()-start,input_sha256=inputs,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'reproduction/boya/asset_adapter.py'):sha(ROOT/'reproduction/boya/asset_adapter.py'),str(ROOT/'reproduction/boya/build_collection_pool.py'):sha(ROOT/'reproduction/boya/build_collection_pool.py')})
for p,h in inputs.items():
 if sha(p)!=h:raise ValueError('Changed input '+p)
Path('/tmp/S40_CONTACT_WIDTH_REFERENCE_PROBE_20261107.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('SUMMARY',report['old_admitted'],report['new_admitted'],'wall_s',report['wall_s'],flush=True)
