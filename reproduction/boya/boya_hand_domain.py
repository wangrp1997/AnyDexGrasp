"""Boya input domain using calibrated openings and exact source-plane support.

Only the TABLE PREFIX of the existing collision predicate is evaluated here.
The visible point-cloud collision test and full arm checks stay after scoring.
No camera direction cone, object identity, semantics or outcome labels.
"""
from __future__ import annotations
from hashlib import sha256
import json
import numpy as np
from scipy.spatial import ConvexHull
from asset_adapter import cgr_reference_pose,map_slot
from boya_prehead_domain import _raw_rows,_camera_transform
from deploy_score_first import make_queries,expand_and_rank


def array_identity(a):
 a=np.ascontiguousarray(a)
 return dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha256(a.tobytes()).hexdigest())


def calibrated_opening(raw_width,slots):
 """Exactly the old per-family1.7/clip/nearest-.002 mapping, including ties."""
 expanded=float(raw_width)*1.7
 lo=min(s['reference_width_m']for s in slots);hi=max(s['reference_width_m']for s in slots)
 requested=float(np.clip(expanded,lo,hi))
 matches=[s for s in slots if abs(s['reference_width_m']-requested)<=.002]
 if not matches:return None
 slot=min(matches,key=lambda s:(abs(s['reference_width_m']-requested),s['slot_id']))
 return slot,dict(requested_opening_m=requested,width_expansion=1.7,width_clamped=bool(expanded<lo or expanded>hi))


def plane_minimum(vertices,palm,approach,pregrasp=.1):
 """Linear trajectory has the smallest plane clearance at one endpoint."""
 low=float(np.min(np.asarray(vertices)@palm[2,:3])+palm[2,3])
 return min(low,float(low-pregrasp*approach[2]))


class SourceHandSupport:
 def __init__(self,assets):self.assets=assets;self.states={};self.hits=0;self.boundary_rechecks=0
 def points(self,slot):
  q=np.asarray(slot['independent_q'],dtype=np.float64);key=q.tobytes()
  if key in self.states:self.hits+=1;return self.states[key]
  frames,_=self.assets.hand.frames(q)
  parts=[mesh.vertices@frames[name][:3,:3].T+frames[name][:3,3]for name,mesh in self.assets.meshes.items()]
  all_points=np.concatenate(parts)
  # A linear plane support is identical on a set and its convex hull. These
  # are a SUBSET of original source vertices, without QJ/jitter or coordinate changes.
  points=all_points[ConvexHull(all_points).vertices].copy()
  entry=dict(points=points,source_vertex_count=len(all_points),hull_vertex_count=len(points),slot_id=slot['slot_id'])
  self.states[key]=entry;return entry
 def exact_boundary_minimum(self,used,camera,slot,extra):
  self.boundary_rechecks+=1
  palm=map_slot(used,camera,slot,insertion_m=.01*extra)
  approach=cgr_reference_pose(used,camera)[:3,2]
  frames,_=self.assets.hand.frames(np.asarray(slot['independent_q'],dtype=np.float64));low=1.e9
  # Preserve the original per-mesh multiplication/addition and min order at
  # the gate boundary; never loosen the .0005 comparison through tolerance.
  for name,mesh in self.assets.meshes.items():
   frame=frames[name];v=mesh.vertices@frame[:3,:3].T+frame[:3,3]
   world=v@palm[:3,:3].T+palm[:3,3]
   low=min(low,float(world[:,2].min()),float(world[:,2].min()-.1*approach[2]))
  return low
 def minima(self,used,camera,reference,slot,clearance=.0005):
  palm=reference@np.linalg.inv(np.asarray(slot['palm_T_reference']))
  approach=reference[:3,2]
  points=self.points(slot)['points'];values=[]
  for extra in range(4):
   shifted=palm.copy();shifted[:3,3]+=.01*extra*approach
   low=plane_minimum(points,shifted,approach)
   if abs(low-clearance)<1.e-10:low=self.exact_boundary_minimum(used,camera,slot,extra)
   values.append(low)
  return values


def validate_policy(policy):
 if policy.get('schema')!='anydex_boya_hand_domain_policy_v1':raise ValueError('Explicit source-bound hand-domain policy required')
 expected=dict(table_height_world_m=0.,table_clearance_m=.0005,pregrasp_distance_m=.1,
  raw_width_bounds_m=[0.,.099],width_expansion=1.7,nearest_slot_tolerance_m=.002,
  width_clamp=True,extra_depth_m=[0.,.01,.02,.03],flip_policy='both')
 if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('Original physical/mapping constants must remain explicit and unchanged')
 box=np.asarray(policy['world_workspace_aabb_m'],dtype=float)
 if box.shape!=(2,3)or not np.isfinite(box).all()or np.any(box[0]>=box[1]):raise ValueError('Explicit sensor AABB required')
 if not policy.get('source_sha256'):raise ValueError('Bound sources required')
 return box


def make_hand_queries(raw,features,camera,policy,profile,families,assets,provenance,support=None):
 raw=_raw_rows(raw,rigid=True);camera=_camera_transform(camera);box=validate_policy(policy)
 if raw.shape[1]!=15 or features.shape!=(len(raw),2261)or not np.isfinite(features).all():raise ValueError('Paired original Nx15/Nx2261 arrays required')
 if set(families)!={1,6,7,8}:raise ValueError('Exactly four calibrated Boya families required')
 support=SourceHandSupport(assets)if support is None else support
 xyz=raw[:,12:15].astype(float)@camera[:3,:3].T+camera[:3,3]
 widths=(raw[:,1]>0)&(raw[:,1]<.099)
 workspace=np.all((xyz>box[0])&(xyz<box[1]),axis=1)
 eligible={};all_records=[];mapped=0;plane_tests=0
 for index in np.flatnonzero(widths&workspace):
  opening={g:calibrated_opening(raw[index,1],slots)for g,slots in families.items()}
  for flipped in (False,True):
   used=raw[index].copy()
   if flipped:used[3:12].reshape(3,3)[:,1:3]*=-1
   reference=cgr_reference_pose(used,camera)
   for gesture in (1,6,7,8):
    if opening[gesture]is None:continue
    slot,metadata=opening[gesture];mapped+=1;plane_tests+=4
    minima=support.minima(used,camera,reference,slot)
    allowed=[extra for extra,low in enumerate(minima)if low>=.0005]
    record=dict(source_candidate_index=int(index),gesture_id=gesture,if_flip=flipped,
      domain_source_slot=slot['slot_id'],allowed_extra_depth_indices=allowed,table_minimum_heights_m=minima,**metadata)
    all_records.append(record)
    if allowed:eligible[(int(index),gesture,flipped)]=record
 indices=np.asarray(sorted({key[0]for key in eligible}),dtype=np.int64)
 queries,inputs,local=make_queries(raw[indices],features[indices],profile,'both')
 selected=[int(indices[i])for i in local];kept_queries=[];kept_inputs=[]
 for query,feature in zip(queries,inputs):
  query['source_candidate_index']=int(indices[query['source_candidate_index']])
  record=eligible.get((query['source_candidate_index'],query['gesture_id'],query['if_flip']))
  if record is None:continue
  query.update(record,query_index=len(kept_queries));kept_queries.append(query);kept_inputs.append(feature)
 receipt=dict(schema='anydex_boya_hand_domain_receipt_v1',raw_input=array_identity(raw),T_WC=camera.tolist(),policy=policy,source_provenance=provenance,
  counts=dict(input=len(raw),raw_width_pass=int(widths.sum()),world_workspace_pass=int(workspace.sum()),
    common_pass=int((widths&workspace).sum()),mapped_pose_queries=mapped,support_plane_tests=plane_tests,
    hand_domain_raw=len(indices),hand_domain_pose_queries=len(eligible),raw_topk=len(selected),head_queries=len(kept_queries),
    allowed_extra_combinations=sum(len(q['allowed_extra_depth_indices'])for q in kept_queries)),
  eligible_source_indices=indices.tolist(),selected_source_indices=selected,opening_support_records=all_records,
  support_cache_states=len(support.states),support_cache_hits=support.hits,boundary_rechecks=support.boundary_rechecks,
  source_hull_reduction='Exact support of source vertex union; only convex-hull vertex indices retained; no jitter/shrink',
  comparisons='Original table gate accepts minimum>=.0005; original raw width/workspace use strict open bounds',
  full_cloud_collision_calls_before_scoring=0,arm_calls_before_scoring=0,labels_read=False)
 return kept_queries,kept_inputs,selected,receipt


def expand_hand_queries(queries,profile):
 """Filter known plane-infeasible EXTRA poses before applying global budgets."""
 all_rows,_=expand_and_rank(queries,profile)
 allowed={q['query_index']:set(q['allowed_extra_depth_indices'])for q in queries}
 rows=[row for row in all_rows if row['extra_depth_index']in allowed[row['query_index']]]
 for rank,row in enumerate(rows):row['global_score_rank']=rank
 shortlist=rows[:profile['score_topk']]
 if profile['strict_score_threshold']is not None:
  threshold=float(np.float32(profile['strict_score_threshold']))
  shortlist=[row for row in shortlist if row['learned_score']>threshold]
 return rows,shortlist[:profile['collision_topk']]
