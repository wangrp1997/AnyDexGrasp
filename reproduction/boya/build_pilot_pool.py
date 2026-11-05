"""Pilot collection geometry with cached per-opening exact source hulls.

Uses point-cloud collision and source hand geometry only. No CanonDex grasp
sampler, experience retrieval, local grasp optimization or learned DH3 head.
The existing full-arm backend must additionally admit each requested path.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull

from asset_adapter import BoyaAssets,map_slot,cgr_reference_pose,sha


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


class OpeningGeometryCache:
    """Reuse only immutable hand FK/hulls; scene and palm checks stay per row."""
    def __init__(self):
        self.states={}
        self.hits=0

    def shapes(self,assets,q):
        key=np.asarray(q,dtype=np.float64).tobytes()
        if key in self.states:
            self.hits+=1
            return self.states[key]
        frames,_=assets.hand.frames(q)
        shapes=[]
        for name,mesh in assets.meshes.items():
            f=frames[name]
            v=mesh.vertices@f[:3,:3].T+f[:3,3]
            h=ConvexHull(v)
            shapes.append((v,v.min(0),v.max(0),h.equations))
        self.states[key]=shapes
        return shapes


def geometry(assets,q,palm,cloud,approach,cache=None):
    cache=cache if cache is not None else OpeningGeometryCache()
    shapes=cache.shapes(assets,q)
    low=1.e9
    for v,a,b,eq in shapes:
        world=v@palm[:3,:3].T+palm[:3,3]
        low=min(low,float(world[:,2].min()),float(world[:,2].min()-.1*approach[2]))
    if low < .0005:return dict(ok=False,reason='table',minimum_height_m=low)
    local=(cloud-palm[:3,3])@palm[:3,:3]
    d=approach@palm[:3,:3]
    for distance in np.linspace(0.,.1,21):
        points=local+distance*d
        for v,a,b,eq in shapes:
            subset=points[np.all((points>=a)&(points<=b),axis=1)]
            if len(subset) and np.any(np.all(subset@eq[:,:3].T+eq[:,3] < -1e-7,axis=1)):
                return dict(ok=False,reason='visible_scene_point_in_hand_hull',approach_distance_m=float(distance))
    return dict(ok=True,minimum_height_m=low,approach_samples=21,
        boundary='Partial visible scene points; original complete-arm source-geometry and native gates still required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--frame-dir',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--gesture-id',type=int,default=1)
    p.add_argument('--candidate-limit',type=int,default=500)
    p.add_argument('--width-expansion',type=float,default=1.7)
    p.add_argument('--opponent',choices=['ff','mf','rf','lf'],default='ff')
    p.add_argument('--keep',type=int,default=8)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if not (np.isfinite(a.width_expansion) and a.width_expansion>0 and a.candidate_limit>0 and a.keep>0):
        p.error('Finite positive opening expansion and positive candidate/keep budgets required')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    obs=json.loads(a.observation.read_text());cal=json.loads(a.calibration.read_text())
    receipt=json.loads((a.frame_dir/'RESULTS.json').read_text())
    if receipt.get('full_representation_forward') is not True:raise ValueError('No actual CGR forward')
    for item in obs['files'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Observation changed')
    source=a.frame_dir/'raw_camera_candidates.npz'
    with np.load(source,allow_pickle=False) as z:raw=z['candidates'].copy();features=z['candidate_features'].copy()
    if len(raw)!=len(features):raise ValueError('Feature row mismatch')
    with np.load(a.frame_dir/'representation_outputs.npz',allow_pickle=False) as z:cloud=z['sparse_input_XYZ_features'].copy()
    camera=np.array(obs['T_WC']);cloud=cloud@camera[:3,:3].T+camera[:3,3]
    assets=BoyaAssets();family=assets.families[a.gesture_id]
    lock=cal['source_lock']
    if lock['independent_motor_names']!=list(assets.hand.names):
        raise ValueError('Calibration motor order differs from this hand')
    for path,record in lock['source_files'].items():
        if sha(path)!=record['sha256']:raise ValueError('Stale calibration source asset: '+path)
    for path,digest in lock['readonly_fk_dependencies'].items():
        if sha(path)!=digest:raise ValueError('Stale calibration FK dependency: '+path)
    slots=[s for s in cal['slots'] if s['gesture_id']==a.gesture_id and s['self_collision_clear'] and s['opposing_group']==a.opponent]
    if not slots:raise ValueError('No calibrated opening family for this declared type datum')
    for slot in slots:
        if slot['wrist']!=family.wrist or not np.array_equal(slot['close_q'],family.nominal):
            raise ValueError('Calibration wrist or closure differs from the declared source family')
        expected=family.motor_state(slot['fraction'],slot['fraction'])
        if not np.array_equal(slot['independent_q'],expected):
            raise ValueError('Calibration opening differs from the declared source family')
    lower=min(s['reference_width_m'] for s in slots);upper=max(s['reference_width_m'] for s in slots)
    records=[];rejected=[];palms=[];qs=[];closed=[];directions=[]
    cache=OpeningGeometryCache()
    for index in np.argsort(-raw[:,0],kind='stable')[:a.candidate_limit]:
        row=raw[index]
        expanded=float(row[1])*a.width_expansion
        requested=float(np.clip(expanded,lower,upper))
        matches=[s for s in slots if abs(s['reference_width_m']-requested)<=.002]
        if not matches:
            rejected.append(dict(source_candidate_index=int(index),reason='outside_calibrated_width_tolerance'));continue
        slot=min(matches,key=lambda s:(abs(s['reference_width_m']-requested),s['slot_id']))
        # A fixed collection gesture/finite aperture table, as in the author
        # protocol. No execution outcomes are consulted to choose this opening.
        for flipped in [False,True]:
            used=row.copy()
            if flipped:used[3:12].reshape(3,3)[:,1:3]*=-1
            palm=map_slot(used,camera,slot)
            approach=cgr_reference_pose(used,camera)[:3,2]
            check=geometry(assets,np.array(slot['independent_q']),palm,cloud,approach,cache)
            meta=dict(source_candidate_index=int(index),if_flip=flipped,source_slot=slot['slot_id'],
                actual_reference_width_m=slot['reference_width_m'],raw_width_m=float(row[1]),
                requested_opening_m=requested,width_expansion=a.width_expansion,width_clamped=bool(expanded<lower or expanded>upper),
                raw_depth_m=float(row[2]),additional_depth_index=0,additional_depth_m=0.,
                cgr_score=float(row[0]),score_source='CGR antipodal score; no Boya learned head',
                jaw_assignment=dict(thumb='th',opponent=slot['opposing_group']),geometry=check)
            if not check['ok']:
                rejected.append(meta);continue
            records.append(meta);palms.append(palm);qs.append(slot['independent_q'])
            closed.append(slot['close_q']);directions.append(approach)
            if len(records)>=a.keep:break
        if len(records)>=a.keep:break
    if not records:
        save(a.output_dir/'no_candidate.json',dict(records=rejected,physics_steps=0,training_label=None,gesture_id=a.gesture_id,
            status='no_hand_geometry_in_declared_budget',source_candidates_sha256=sha(source),
            source_observation_sha256=sha(a.observation),calibration_sha256=sha(a.calibration),
            elapsed_wall_s=time.monotonic()-start,opening_states_cached=len(cache.states),cache_hits=cache.hits))
        raise RuntimeError('No point-cloud/table-admitted Boya geometry in the declared candidate budget')
    target=a.output_dir/'candidates.npz'
    np.savez_compressed(target,independent_motor_names=assets.hand.names,proposal_q=qs,
        proposal_q_close=closed,proposal_palm=palms,proposal_approach_direction=directions,
        world_T_object=obs['world_T_object'])
    files=[source,a.frame_dir/'RESULTS.json',a.frame_dir/'representation_outputs.npz',a.observation,a.calibration,
        assets.urdf,assets.scene,assets.package,assets.reference_path,Path(__file__),Path(__file__).with_name('asset_adapter.py')]
    object_root=Path(obs['object_root'])
    files.extend([object_root/'mesh/simplified.obj',object_root/'urdf/coacd.urdf'])
    import xml.etree.ElementTree as ET
    for node in ET.parse(object_root/'urdf/coacd.urdf').findall('.//mesh'):
        files.append((object_root/'urdf'/node.get('filename')).resolve())
    files.extend(Path(item['path']) for item in obs['files'].values())
    save(a.output_dir/'generation.json',dict(complete=True,kind='anydex_boya_collection_geometry_v1',
        gesture_id=a.gesture_id,scale=obs['object_scale'],selected_indices=list(range(len(records))),
        candidates_sha256=sha(target),input_sha256={str(f.resolve()):sha(f) for f in files},
        records=records,rejected=rejected,source_frame=str(a.frame_dir.resolve()),
        source_observation=str(a.observation.resolve()),hand_head_loaded=False,
        original_CGR_parser=receipt['candidate_parser_overlay'],physics_steps=0,training_labels=0,
        collection_policy='Fixed declared gesture/datum; CGR score order, author 1.7 opening expansion and explicit hand-table width clamp; two-sided flip; finite geometric admission only',
        object_pose_as_network_input=False,opening_geometry_cache=dict(states=len(cache.states),hits=cache.hits,
            reused='Only local FK and source hull equations; per-candidate table/cloud/approach recomputed'),
        elapsed_wall_s=time.monotonic()-start))
    save(a.output_dir/'world_object_pose.json',obs['world_T_object'])
    print(json.dumps(dict(geometry_admitted=len(records),rejected=len(rejected),elapsed_s=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
